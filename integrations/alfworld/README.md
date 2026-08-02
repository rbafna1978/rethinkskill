# ALFWorld integration

This optional package connects RethinkSkill to an installed ALFWorld `0.4.2`
text environment. It owns dataset loading, bounded interaction, and terminal
success evaluation.

```bash
python -m pip install ./integrations/alfworld
export RETHINKSKILL_ALFWORLD_ASSET_ROOT=/path/to/alfworld_data
rethinkskill --load-harness-plugins benchmark-catalog
```

Dataset rows contain an ID and a corpus-relative game path:

```json
{"id":"alfworld-task-1","gamefile":"json_2.1.1/valid_unseen/task/game.tw-pddl"}
```

The installed environment's terminal `won` state is authoritative, so
ALFWorld is evaluated by the harness rather than a frozen-response scorer.
The package does not download or redistribute the ALFWorld corpus.
