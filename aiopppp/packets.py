import datetime
import json
import logging
import struct

from .const import CAM_MAGIC, CC_DEST, BinaryCommands, PacketType
from .types import Channel, DeviceID

logger = logging.getLogger(__name__)

class Packet:
    def __init__(self, typ, payload):
        self.type = typ
        self._payload = payload

    def get_payload(self):
        return self._payload

    def __str__(self):
        return f'{self.type.name}: [{self.get_payload().hex(" ")}]'

    def __bytes__(self):
        payload = self.get_payload()
        return struct.pack('>BBH', CAM_MAGIC, self.type.value, len(payload)) + payload


class PunchPkt(Packet):
    def __str__(self):
        return f'{self.type.name}: [{self.as_object()}]'

    def as_object(self):
        payload = self.get_payload()
        return DeviceID(
            prefix=payload[:4].decode('ascii'),
            serial=str(struct.unpack('>Q', payload[4:12])[0]),
            suffix=payload[12:].rstrip(b'\x00').decode('ascii'),
        )


class DrwPkt(Packet):
    def __init__(self, channel, cmd_idx, drw_payload):
        super().__init__(PacketType.Drw, None)
        self._channel = Channel(channel)
        self._cmd_idx = cmd_idx
        self._payload = drw_payload

    def get_drw_payload(self):
        return self._payload

    def get_payload(self):
        return struct.pack('>BBH', 0xd1, self._channel.value, self._cmd_idx) + self.get_drw_payload()

    def drw_str(self):
        return f'chn:{self._channel.name}, idx: {self._cmd_idx}'

    def __str__(self):
        # return f'{self.type.name}({self.drw_str()}): [{self._payload.hex(" ")}]'
        return f'{self.type.name}({self.drw_str()}): len={len(self._payload)}]'


class JsonCmdPkt(DrwPkt):
    def __init__(self, cmd_idx, json_payload, preamble=b'\x06\x0a\xa0\x80'):
        super().__init__(0, cmd_idx, None)
        self.json_payload = json_payload
        self.preamble = preamble

    def __str__(self):
        return f'{self.type.name}({self.drw_str()}): [{hex(self.preamble[2])}, {self.json_payload}]'

    def get_drw_payload(self):
        payload = json.dumps(self.json_payload).encode('utf-8')
        return self.preamble + len(payload).to_bytes(4, 'little') + payload


def xq_bytes_encode(data, shift):
    new_buf = bytes(b - 1 if b & 1 else b + 1 for b in data)
    if not new_buf:
        return b''
    # The rotation is modulo the buffer length; a raw shift larger than the
    # payload (e.g. shift=4 on a 1-3 byte payload) would otherwise rotate by the
    # wrong amount and fail to round-trip with xq_bytes_decode.
    shift %= len(new_buf)
    return bytes(new_buf[shift:] + new_buf[:shift])


def xq_bytes_decode(data, shift):
    new_buf = bytes(b - 1 if b & 1 else b + 1 for b in data)
    if not new_buf:
        return b''
    shift %= len(new_buf)
    return bytes(new_buf[-shift:] + new_buf[:-shift])

def _inet_btoa(b: bytes) -> str:
    """
    Convert IP Address from byte array to a dot-separated string.
    
    """
    return '.'.join(str(x) for x in b)

def _get_dev_version(b: bytes) -> str:
    """
    Convert 4-byte version number to a string.
    """
    return '.'.join(str(x) for x in reversed(b))

def parse_dev_status(data):
    """
    Example data (len=124):

    "0d 02 01 3d 74 0f 00 00 00 00 00 00 ff ff ff ff bf ff ff ff "
    "01 01 00 30 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 "
    "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 "
    "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 "
    "00 00 00 00 00 00 00 00 00 01 00 00 02 00 00 00 00 00 00 00 "
    "00 00 00 00 00 ff ff ff 00 00 00 00 ff ff ff ff 00 00 00 00 "
    "00 00 00 00"

    Values are unknown so data below is something that looks similar according to the value ranges.

    """

    logger.debug('Parse dev status [%s]', data.hex(' '))

    if len(data) < 124:
        return {}

    (
        sw_ver,              # 0-3 (4 bytes)
        bat_level,           # 4-7 (int)
        time_zone,           # 8-11 (int)
        rec_nmb,             # 12-15 (int)
        sys_uptime,          # 16-19 (int)
        power_supply,        # 20-23 (int)
        dev_name,            # 24-87 (64 bytes)
        sd_status,           # 88 (1 byte)
        p2p_status,          # 89 (1 byte)
        conn_type,           # 90 (1 byte)
        rec_enable_on_start, # 91 (1 byte)
        pic_enable_on_start, # 92 (1 byte)
        ir_cut,              # 93 (1 byte)
        osd_enable,          # 94 (1 byte)
        alarm_enable,        # 95 (1 byte)
        mode,                # 96 (1 byte)
        dhcp,                # 97 (1 byte)
        mac,                 # 98-103 (6 bytes)
        ip_addr_bytes,       # 104-107 (4 bytes)
        netmask_bytes,       # 108-111 (4 bytes)
        pic_nmb,             # 112-115 (int)
        total_size,          # 116-119 (int)
        used_size            # 120-123 (int)
    ) = struct.unpack('<4s5i64s10B6s4s4s3I', data[:124])

    # time_zone is in seconds WEST of UTC (UTC+2 is stored as -7200, confirmed
    # on PTZA hardware). Firmwares without a timezone leave a constant here, so
    # only render a zone when the value is actually plausible.
    utc_offset = _tz_west_seconds(time_zone)
    return {
        'tz': f'UTC{utc_offset // 3600:+d}' if utc_offset is not None else None,
        'utcOffsetSeconds': utc_offset,
        'uptime': sys_uptime,
        'uptimeText': _fmt_uptime(sys_uptime),
        # Real Wi-Fi signal strength is not identified in this 124-byte struct;
        # don't masquerade the uptime as dBm (it produced bogus signal readings).
        'dbm': None,
        'devName': dev_name.decode('ascii', errors='ignore').rstrip('\0'),
        'sdStatus': sd_status,
        'p2pStatus': p2p_status,
        'connType': conn_type,
        'osdEnable': osd_enable,
        'alarmEnable': alarm_enable,
        'mode': mode,
        'recEnableOnStart': rec_enable_on_start,
        'picEnableOnStart': pic_enable_on_start,
        'recNmb': rec_nmb,
        'picNmb': pic_nmb,
        'totalSize': total_size,
        'usedSize': used_size,
        # Only bit 0 of powerSupply is meaningful (external power vs battery)
        # -- the vendor app displays getPowerSupply() & 1; the high bytes are
        # unrelated flags. batLevel is the battery voltage in millivolts.
        'powerSupply': power_supply,
        'externalPower': bool(power_supply & 1),
        'batLevel': bat_level,
        'batPercent': _bat_percent(bat_level),
        'dhcp': dhcp,
        'ipAddr': _inet_btoa(ip_addr_bytes),
        'netmask': _inet_btoa(netmask_bytes),
        'mac':mac.hex(':'),
        'mcuver': _get_dev_version(sw_ver),
        'icut': ir_cut,
        'lamp': 0, # lamp is not in the status
    }


def _cstr(b: bytes) -> str:
    return b.split(b'\x00', 1)[0].decode('utf-8', errors='replace')


def _bat_percent(mv):
    """Battery millivolts -> percent bucket, using the vendor app's own
    thresholds (FtyCamPro batImgGet). None outside the plausible LiPo range
    (externally powered cameras report values like 8000)."""
    if not 3000 <= mv <= 4600:
        return None
    for limit, pct in ((4350, 100), (4200, 80), (4100, 60), (3950, 40), (3900, 20)):
        if mv >= limit:
            return pct
    return 10


def _fmt_uptime(seconds):
    """Uptime seconds -> '1d 2h 3m'. None for negative/garbage values (the
    vendor app never displays this field; some firmwares report junk)."""
    if seconds < 0:
        return None
    minutes, _ = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days:
        parts.append(f'{days}d')
    if hours or days:
        parts.append(f'{hours}h')
    parts.append(f'{minutes}m')
    return ' '.join(parts)


def _render_ts(ts):
    return datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def _tz_west_seconds(value):
    """Return the UTC offset (seconds EAST) for a raw 'seconds west of UTC'
    field, or None when the value can't be a timezone.

    Firmwares that don't store a timezone leave a constant in this field
    (FTYC reports 224), which naively rendered as a bogus zone like UTC-1.
    A real timezone is a whole number of 15-minute steps within +/-14 h."""
    if value % 900 or abs(value) > 14 * 3600:
        return None
    return -value


def parse_datetime_block(data):
    """Decode a CMD_SYSTEM_DATETIME_GET response (80 bytes, two firmware
    variants confirmed on hardware):

    - PTZA: u32 unix timestamp (UTC), i32 timezone as seconds WEST of UTC
      (UTC+2 stored as -7200), 8 pad bytes, char ntp_server[64].
    - FTYC: u32 timestamp that already renders as LOCAL time (the camera adds
      its own internally-stored offset when the clock is set), then constant
      non-tz fields, ntp_server at the same offset. There is no tz in the
      block, so the field-4 value (e.g. 0xE0) must not be shown as one.

    The variants are told apart by tz plausibility: a real timezone is a
    multiple of 15 minutes within +/-14 h."""
    if len(data) < 8:
        return {}
    ts, field4 = struct.unpack_from('<Ii', data)
    utc_offset = _tz_west_seconds(field4)
    if utc_offset is not None:
        result = {
            'layout': 'utc+tz',
            'timestamp': ts,
            'utc': _render_ts(ts),
            'local': _render_ts(ts + utc_offset),
            'tz': f'UTC{utc_offset // 3600:+d}',
            'utcOffsetSeconds': utc_offset,
        }
    else:
        result = {
            'layout': 'local-only',
            'timestamp': ts,
            'local': _render_ts(ts),
            'tz': 'device-managed',
        }
    if len(data) >= 80:
        result['ntpServer'] = _cstr(data[16:80])
    return result


def parse_user_block(data):
    """Decode a CMD_SYSTEM_USER_GET response. Layout from the vendor app's
    IpcByte2ObjectParser.ParseUser (minus its 4-byte JNI prefix):
    char account[32], char password[128]."""
    if len(data) < 160:
        return {}
    return {
        'account': _cstr(data[0:32]),
        'password': _cstr(data[32:160]),
    }


def parse_wifi_settings(data):
    """Decode a CMD_NET_WIFISETTING_GET response (layout confirmed on PTZA
    hardware, len=264): u32 mode, 12 pad bytes, u32 security, 4 pad bytes,
    char ssid[32], char password[128], then five char[16] dotted-quad strings
    (ip, netmask, gateway, dns1, dns2)."""
    if len(data) < 184:
        return {}
    mode, = struct.unpack_from('<I', data, 0)
    security, = struct.unpack_from('<I', data, 16)
    result = {
        'mode': mode,
        'security': security,
        'ssid': _cstr(data[24:56]),
        'password': _cstr(data[56:184]),
    }
    for i, key in enumerate(('ip', 'netmask', 'gateway', 'dns1', 'dns2')):
        off = 184 + i * 16
        if len(data) >= off + 16:
            result[key] = _cstr(data[off:off + 16])
    return result


class BinaryCmdPkt(DrwPkt):
    START_CMD = b'\x11\x0a'
    HEADER_FORMAT = '<2s3H'

    def __init__(self, cmd_idx, command, cmd_payload, token=b'\x00\x00\x00\x00'):
        super().__init__(0, cmd_idx, None)
        self.command = command
        self.cmd_payload = cmd_payload
        # don't know what is token, but it comes in the beginning of the payload
        # for BATE camera it is always 0x00000000
        self.token = token

    def __str__(self):
        return f'{self.type.name}({self.drw_str()}): {self.command}, (token: {self.token.hex()}) [{self.cmd_payload}]'

    def get_drw_payload(self):
        data = struct.pack(
            self.HEADER_FORMAT,
            self.START_CMD,
            self.command.value,
            len(self.cmd_payload) + len(self.token),
            CC_DEST.get(self.command, 0x0),
        )
        data += self.token
        if self.cmd_payload:
            data += xq_bytes_encode(self.cmd_payload, 4)
        return data

def pack_passtrough_cmd(command, data):
    START_CMD = 0x010A
    HEADER_FORMAT = '>4H4x' # Four 2-byte unsigned shorts, and 4 bytes padding (4x)
    CMD_DEST = 0xFFFF
    SHORT_MASK = 0xFFFF

    header = struct.pack(HEADER_FORMAT, START_CMD, command & SHORT_MASK, (len(data) + 4) & SHORT_MASK, CMD_DEST)
    length = struct.pack('<I', len(header) + len(data))
    return length + header + data

def parse_punch_pkt(data):
    return PunchPkt(PacketType.PunchPkt, data)


def parse_p2prdy_pkt(data):
    return PunchPkt(PacketType.P2pRdy, data)


def make_punch_pkt(dev_id):
    return PunchPkt(
        PacketType.PunchPkt,
        struct.pack(
            '>4sQ8s',
            dev_id.prefix.encode('ascii'),
            int(dev_id.serial),
            dev_id.suffix.encode('ascii'),
        )
    )


def parse_drw_pkt(data):
    channel, cmd_idx = struct.unpack('>xBH', data[:4])
    if data[4:6] == b'\x06\x0a':
        try:
            return JsonCmdPkt(cmd_idx, json.loads(data[12:]), preamble=data[4:8])
        except ValueError:
            logging.warning(f'Failed to parse JSON: {data}')
    elif data[4:6] == b'\x11\x0a':
        try:
            _, command_num, length, dest = struct.unpack(BinaryCmdPkt.HEADER_FORMAT, data[4:12])
            cmd_bin_payload = data[12:]
            token = b'\x00\x00\x00\x00'
            if len(cmd_bin_payload) < 4:
                logging.warning('Binary command payload too short: [%s]', cmd_bin_payload.hex(' '))
            else:
                # assume first 4 bytes is token and other part - xq_encoded payload
                token, cmd_bin_payload = cmd_bin_payload[:4], cmd_bin_payload[4:]
                if len(cmd_bin_payload):
                    cmd_bin_payload = xq_bytes_decode(cmd_bin_payload, 4)
            pkt = BinaryCmdPkt(
                cmd_idx=cmd_idx,
                command=BinaryCommands(command_num),
                token=token,
                cmd_payload=cmd_bin_payload,
            )
            logger.debug('Parsed binary command: %s, raw=[%s]', pkt, data.hex(" "))
            return pkt
        except ValueError:
            logging.warning(f'Failed to parse binary command: {data}')
    return DrwPkt(channel, cmd_idx, data[4:])


def make_audio_drw_pkt(cmd_idx, payload):
    """Outgoing audio (talk-back) frame on the audio DRW channel."""
    return DrwPkt(Channel.Audio, cmd_idx, payload)


def make_drw_ack_pkt(drw_pkt):
    return Packet(
        PacketType.DrwAck,
        struct.pack('>BBHH', 0xd1, drw_pkt._channel.value, 1, drw_pkt._cmd_idx)
    )


def make_p2palive_pkt():
    return Packet(PacketType.P2PAlive, b'')


def make_p2palive_ack_pkt():
    return Packet(PacketType.P2PAliveAck, b'')


def make_close_pkt():
    return Packet(PacketType.Close, b'')


PARSERS = {
    PacketType.PunchPkt: (PunchPkt, parse_punch_pkt),
    PacketType.P2pRdy: (PunchPkt, parse_p2prdy_pkt),
    PacketType.Drw: (DrwPkt, parse_drw_pkt),
}


def parse_packet(data):
    if data[0] != CAM_MAGIC:
        raise ValueError('Invalid data')

    typ, length = struct.unpack('>xBH', data[:4])
    if len(data) != length + 4:
        # some cameras are known to send broken p2p alive packets - zero length, but real length is different and
        # payload consists of zeros
        if typ == PacketType.P2PAlive.value:
            data = data[:4]
        else:
            logger.debug(
                'Invalid pkt length: pkt.len=%d, real length=%d, [%s]',
                length, len(data) - 4, data.hex(' '))

    try:
        packet_type = PacketType(typ)
    except ValueError:
        # A corrupt or unrecognized datagram must not raise out of the UDP
        # receive callback; surface it as ValueError so callers drop it.
        raise ValueError(f'Unknown packet type 0x{typ:02x}')

    pkt_class, parse_func = PARSERS.get(packet_type, (Packet, None))
    if parse_func is None:
        return pkt_class(packet_type, data[4:])
    return parse_func(data[4:])
