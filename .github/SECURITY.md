# RethinkSkill security policy

## Reporting a vulnerability

Report security issues privately through the repository's GitHub security
advisory interface. Do not open a public issue containing credentials,
provider responses, private benchmark data, raw trajectories, receipts, or
other protected evidence.

Include the affected version, a minimal zero-model reproducer when possible,
the expected behavior, and the observed behavior. Remove tokens, hostnames,
user-specific paths, and benchmark content from diagnostic output.

## Credential handling

Credentials must enter the runtime only through the explicitly selected
provider configuration. They must not be committed, serialized into plans or
receipts, passed to benchmark plugins, or inherited by unrelated child
processes.

If a credential is exposed, revoke or rotate it immediately. Removing it from
a later commit does not remove it from Git history.

## Supported versions

Security fixes are applied to the latest release on the default branch. This
research repository does not promise long-term support for older releases.
