# IPython MCP sensitivity control

The main `ipython` condition executes the unmodified
[gabiteodoru/ipython-mcp](https://github.com/gabiteodoru/ipython-mcp) server at
`c2fa8d6fdafe15d7ebbaebb2d32f2e41882227d0`. Its `server.py` SHA-256 is
`aa1bf011536c1a79d31f510e07175005d05eaad26f1fc2bbe75a0c8aa10bce2a`.

The separate `ipython_fixed` control applies `ipython-reply-correlation.patch`
to another checkout of that exact revision. It makes two changes:

- Wait for the shell reply whose parent message ID matches the current code
  request, discarding late replies from previous timed-out calls.
- Use a 300-second execution wait, configurable through
  `IPYTHON_MCP_EXECUTION_TIMEOUT`, instead of the hardcoded 30 seconds. A
  timeout explicitly reports that the code may still be running.

No simulator code, tool list, Python namespace, physics, scoring, or agent
budget changes. The benchmark records the actual corrected entry file hash
separately from the installed upstream package. This is a disclosed local
comparison control, not an upstream release or contribution.

Apply the patch using `git apply /path/to/ipython-reply-correlation.patch` in
the second checkout. Set `NEWTON_EVAL_IPYTHON_FIXED_SERVER` to that checkout's
`ipython_mcp/server.py` before selecting `--condition ipython_fixed`.

The upstream code is MIT-licensed; its attribution and license are retained
in `IPYTHON_LICENSE`. It remains a separately installed evaluation dependency,
not a required Newton dependency.
