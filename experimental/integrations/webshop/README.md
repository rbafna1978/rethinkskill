# Experimental WebShop integration

This optional external package connects RethinkSkill to a separately installed
official WebShop text environment through a bounded JSONL subprocess bridge.
It is not part of the six-benchmark paper release.

Required local values:

```bash
export RETHINKSKILL_WEBSHOP_ROOT=/path/to/WebShop
export RETHINKSKILL_WEBSHOP_PYTHON=/path/to/webshop-python
export JAVA_HOME=/path/to/jdk
python -m pip install ./experimental/integrations/webshop --no-deps
rethinkskill --load-harness-plugins benchmark-catalog
```

A dataset row identifies an official shuffled goal:

```json
{"id":"webshop-0","session":0,"max_steps":15,"num_products":1000}
```

The adapter records official observations, actions, reward, and termination.
It does not download WebShop, product data, goals, or a search index. Runtime
readiness therefore remains false until the separately installed environment
passes preflight.
