from __future__ import annotations

import socket
import struct
import uuid
from typing import Iterable, Tuple


class SacnSender:
    _PORT = 5568
    _DMX_CHANNELS_PER_UNIVERSE = 512
    _MAX_WLED_UNIVERSES = 9

    def __init__(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cid = uuid.uuid4().bytes
        self._sequence = 0

    def close(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass

    def send_pixels(
        self,
        destination_ip: str,
        universe: int,
        start_address: int,
        pixels: Iterable[Tuple[int, int, int]],
        source_name: str = "HS Pose",
    ) -> None:
        ip = (destination_ip or "").strip()
        if not ip:
            return

        universe = max(1, min(63999, int(universe)))
        start_address = max(1, min(512, int(start_address)))

        base_universe = universe
        universe_offset = 0
        dmx_slots = [0] * self._DMX_CHANNELS_PER_UNIVERSE
        channel = start_address - 1

        for red, green, blue in pixels:
            if universe_offset >= self._MAX_WLED_UNIVERSES:
                break

            if channel + 2 >= self._DMX_CHANNELS_PER_UNIVERSE:
                self._send_universe_packet(
                    ip=ip,
                    universe=base_universe + universe_offset,
                    dmx_slots=dmx_slots,
                    source_name=source_name,
                )
                universe_offset += 1
                if universe_offset >= self._MAX_WLED_UNIVERSES:
                    break
                dmx_slots = [0] * self._DMX_CHANNELS_PER_UNIVERSE
                channel = 0

            dmx_slots[channel] = max(0, min(255, int(red)))
            dmx_slots[channel + 1] = max(0, min(255, int(green)))
            dmx_slots[channel + 2] = max(0, min(255, int(blue)))
            channel += 3

        if universe_offset < self._MAX_WLED_UNIVERSES:
            self._send_universe_packet(
                ip=ip,
                universe=base_universe + universe_offset,
                dmx_slots=dmx_slots,
                source_name=source_name,
            )

    def _send_universe_packet(
        self,
        ip: str,
        universe: int,
        dmx_slots: list[int],
        source_name: str,
    ) -> None:
        packet = self._build_packet(universe, dmx_slots, source_name)
        self._socket.sendto(packet, (ip, self._PORT))

    def _build_packet(self, universe: int, dmx_slots: list[int], source_name: str) -> bytes:
        self._sequence = (self._sequence + 1) % 256
        properties = bytes([0]) + bytes(dmx_slots)

        dmp_pdu = bytearray()
        dmp_pdu.extend(self._flags_and_length(10 + len(properties)))
        dmp_pdu.extend(struct.pack("!B", 0x02))
        dmp_pdu.extend(struct.pack("!B", 0xA1))
        dmp_pdu.extend(struct.pack("!H", 0x0000))
        dmp_pdu.extend(struct.pack("!H", 0x0001))
        dmp_pdu.extend(struct.pack("!H", len(properties)))
        dmp_pdu.extend(properties)

        framing_pdu = bytearray()
        framing_pdu.extend(self._flags_and_length(77 + len(dmp_pdu)))
        framing_pdu.extend(struct.pack("!I", 0x00000002))
        encoded_name = source_name.encode("utf-8", errors="ignore")[:64]
        framing_pdu.extend(encoded_name + b"\x00" * (64 - len(encoded_name)))
        framing_pdu.extend(struct.pack("!B", 100))
        framing_pdu.extend(struct.pack("!H", 0))
        framing_pdu.extend(struct.pack("!B", self._sequence))
        framing_pdu.extend(struct.pack("!B", 0))
        framing_pdu.extend(struct.pack("!H", universe))
        framing_pdu.extend(dmp_pdu)

        root_pdu = bytearray()
        root_pdu.extend(struct.pack("!H", 0x0010))
        root_pdu.extend(struct.pack("!H", 0x0000))
        root_pdu.extend(b"ASC-E1.17\x00\x00\x00")
        root_pdu.extend(self._flags_and_length(22 + len(framing_pdu)))
        root_pdu.extend(struct.pack("!I", 0x00000004))
        root_pdu.extend(self._cid)
        root_pdu.extend(framing_pdu)
        return bytes(root_pdu)

    @staticmethod
    def _flags_and_length(length_value: int) -> bytes:
        value = 0x7000 | (length_value & 0x0FFF)
        return struct.pack("!H", value)
