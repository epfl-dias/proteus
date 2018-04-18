import socket
import argparse

host = "localhost"


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--socket',
		action="store_true",
		help="Set to use socket for communication")
	parser.add_argument('--port',
		action="store",
		help="Set localhost port to use",
		default=50001,
		ype=int)

	args = parser.parse_args()

	if(args.socket):