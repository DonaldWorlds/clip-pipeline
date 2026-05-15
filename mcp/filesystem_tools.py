import os


def list_renders():
    return os.listdir("renders")


def read_log_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def list_recordings():
    return os.listdir("recordings")

'''
The actual Model Context Protocol
 is a STANDARD for exposing tools/resources to AI models in a clean unified way.
'''


'''
ONLY these:

1. Filesystem MCP

Most important.

Lets AI:

inspect clips
read logs
see renders
analyze outputs

3. Terminal MCP

Critical.

Lets AI:

run ffmpeg
run ffprobe
inspect outputs
debug errors

'''