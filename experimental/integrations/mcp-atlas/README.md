# MCP-Atlas external boundary

MCP-Atlas is listed in the benchmark catalog so that it can be selected by
future extensions, but this directory intentionally contains no runnable
adapter or scorer. The official multi-server environment, credentials, agent
harness, and judge remain external to this repository.

The capability is therefore reported as `external_only`. A future independent
package must preserve the official execution and evaluation authority and
provide freezeable trajectories and call accounting before it can register a
runnable harness.

- [Official repository](https://github.com/scaleapi/mcp-atlas)
- [Paper](https://arxiv.org/abs/2602.00933)
