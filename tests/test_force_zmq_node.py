import unittest

from nodes.force_zmq_node import parse_args, publish_values


class FakeSensor:
    def __init__(self):
        self.reads = iter(([1.25, -2], []))

    def read_buf_data(self):
        return next(self.reads)


class FakePublisher:
    def __init__(self):
        self.messages = []

    def send_string(self, message):
        self.messages.append(message)


class ForceZmqNodeTests(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = parse_args([])
        self.assertEqual(args.port, "/dev/ttyUSB0")
        self.assertEqual(args.baudrate, 2400)
        self.assertEqual(args.endpoint, "tcp://127.0.0.1:5577")

    def test_publish_values_sends_plain_numbers(self):
        sensor = FakeSensor()
        publisher = FakePublisher()
        calls = iter((True, True, False))

        publish_values(sensor, publisher, 0, lambda: next(calls))

        self.assertEqual(publisher.messages, ["1.25", "-2.0"])


if __name__ == "__main__":
    unittest.main()
