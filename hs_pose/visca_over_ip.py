import socket


class ViscaOverIpClient:
    def __init__(self, host: str, port: int = 52381, timeout: float = 0.5) -> None:
        self.host = host.strip()
        self.port = int(port)
        self.timeout = float(timeout)
        self._sequence = 1

    def pan_tilt(self, pan_speed: int, tilt_speed: int, pan_dir: int, tilt_dir: int) -> None:
        speed_pan = max(1, min(0x18, int(pan_speed)))
        speed_tilt = max(1, min(0x14, int(tilt_speed)))
        payload = bytes(
            [0x81, 0x01, 0x06, 0x01, speed_pan, speed_tilt, pan_dir, tilt_dir, 0xFF]
        )
        self._send_command(payload)

    def pan_tilt_stop(self, pan_speed: int = 0x08, tilt_speed: int = 0x08) -> None:
        self.pan_tilt(pan_speed=pan_speed, tilt_speed=tilt_speed, pan_dir=0x03, tilt_dir=0x03)

    def pan_left(self, pan_speed: int = 0x08, tilt_speed: int = 0x08) -> None:
        self.pan_tilt(pan_speed=pan_speed, tilt_speed=tilt_speed, pan_dir=0x01, tilt_dir=0x03)

    def pan_right(self, pan_speed: int = 0x08, tilt_speed: int = 0x08) -> None:
        self.pan_tilt(pan_speed=pan_speed, tilt_speed=tilt_speed, pan_dir=0x02, tilt_dir=0x03)

    def tilt_up(self, pan_speed: int = 0x08, tilt_speed: int = 0x08) -> None:
        self.pan_tilt(pan_speed=pan_speed, tilt_speed=tilt_speed, pan_dir=0x03, tilt_dir=0x01)

    def tilt_down(self, pan_speed: int = 0x08, tilt_speed: int = 0x08) -> None:
        self.pan_tilt(pan_speed=pan_speed, tilt_speed=tilt_speed, pan_dir=0x03, tilt_dir=0x02)

    def zoom(self, speed: int, direction: str) -> None:
        zoom_speed = max(0, min(7, int(speed)))
        if direction == "in":
            code = 0x20 + zoom_speed
        elif direction == "out":
            code = 0x30 + zoom_speed
        else:
            code = 0x00
        self._send_command(bytes([0x81, 0x01, 0x04, 0x07, code, 0xFF]))

    def zoom_in(self, speed: int = 2) -> None:
        self.zoom(speed=speed, direction="in")

    def zoom_out(self, speed: int = 2) -> None:
        self.zoom(speed=speed, direction="out")

    def zoom_stop(self) -> None:
        self.zoom(speed=0, direction="stop")

    def focus_far(self) -> None:
        self._send_command(bytes([0x81, 0x01, 0x04, 0x08, 0x02, 0xFF]))

    def focus_near(self) -> None:
        self._send_command(bytes([0x81, 0x01, 0x04, 0x08, 0x03, 0xFF]))

    def focus_stop(self) -> None:
        self._send_command(bytes([0x81, 0x01, 0x04, 0x08, 0x00, 0xFF]))

    def autofocus_on(self) -> None:
        self._send_command(bytes([0x81, 0x01, 0x04, 0x38, 0x02, 0xFF]))

    def home(self) -> None:
        self._send_command(bytes([0x81, 0x01, 0x06, 0x04, 0xFF]))

    def inquiry_power(self) -> bool | None:
        payload = self._send_inquiry(bytes([0x81, 0x09, 0x04, 0x00, 0xFF]))
        if len(payload) >= 4 and payload[0] == 0x90 and payload[1] == 0x50 and payload[-1] == 0xFF:
            if payload[2] == 0x02:
                return True
            if payload[2] == 0x03:
                return False
        return None

    def inquiry_focus_mode(self) -> str | None:
        payload = self._send_inquiry(bytes([0x81, 0x09, 0x04, 0x38, 0xFF]))
        if len(payload) >= 4 and payload[0] == 0x90 and payload[1] == 0x50 and payload[-1] == 0xFF:
            if payload[2] == 0x02:
                return "Auto"
            if payload[2] == 0x03:
                return "Manual"
        return None

    def inquiry_zoom_position(self) -> int | None:
        payload = self._send_inquiry(bytes([0x81, 0x09, 0x04, 0x47, 0xFF]))
        if len(payload) >= 7 and payload[0] == 0x90 and payload[1] == 0x50 and payload[-1] == 0xFF:
            digits = payload[2:6]
            value = 0
            for d in digits:
                if d > 0x0F:
                    return None
                value = (value << 4) | d
            return value
        return None

    def read_status(self) -> dict:
        return {
            "power": self.inquiry_power(),
            "focus_mode": self.inquiry_focus_mode(),
            "zoom_position": self.inquiry_zoom_position(),
        }

    def _send_command(self, payload: bytes) -> None:
        self._send_packet(payload, expect_response=False)

    def _send_inquiry(self, payload: bytes) -> bytes:
        response = self._send_packet(payload, expect_response=True)
        if response is None:
            raise TimeoutError("No VISCA response received.")
        return response

    def _send_packet(self, payload: bytes, expect_response: bool) -> bytes | None:
        if not self.host:
            raise ValueError("VISCA host is empty.")

        payload_len = len(payload)
        header = bytes(
            [
                0x01,
                0x00,
                (payload_len >> 8) & 0xFF,
                payload_len & 0xFF,
                (self._sequence >> 24) & 0xFF,
                (self._sequence >> 16) & 0xFF,
                (self._sequence >> 8) & 0xFF,
                self._sequence & 0xFF,
            ]
        )
        packet = header + payload
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        if self._sequence == 0:
            self._sequence = 1

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(self.timeout)
            sock.sendto(packet, (self.host, self.port))
            if not expect_response:
                return None
            data, _ = sock.recvfrom(2048)
            if len(data) < 8:
                return b""
            return data[8:]
