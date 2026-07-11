## Context

The CLI already has the pieces needed to source CCP4 setup scripts, but the experience is still too manual. The desired behavior is to fold these checks into the start of the main pipeline so that a normal run can proceed when CCP4 is already configured, and otherwise prompt the user once to provide a setup path that is remembered for later runs.

## Goals / Non-Goals

**Goals:**
- Perform CCP4 readiness checks at the beginning of the main pipeline.
- Resolve CCP4 in a predictable order: CLI override, PATH, environment variable, saved config, and common install locations.
- Prompt the user for a setup path when CCP4 is unavailable and no saved configuration exists, then persist that value for future runs.
- Exit with clear guidance when CCP4 still cannot be resolved.

**Non-Goals:**
- Replacing the core analysis pipeline or changing the underlying CCP4 toolchain.
- Supporting arbitrary shell initialization beyond the existing setup-script model.

## Decisions

- Keep CCP4 detection and persistence logic in [src/ccp4_setup.py](src/ccp4_setup.py) so the startup flow and tests share one implementation.
- Run the CCP4 check before any processing begins in [src/main.py](src/main.py), so startup failures happen immediately and clearly.
- Use JSON config files under repo-local and user-level locations to store a previously provided CCP4 setup path.
- Preserve the explicit `--ccp4-setup` override so advanced users can force a specific setup script when needed.
- When no CCP4 tools are available and no saved path is found, prompt the user once and save the value rather than requiring a repeated manual setup step.

## Risks / Trade-offs

- [A user may provide an invalid setup path] → The flow should validate the path and re-prompt or fail with a clear message instead of silently continuing.
- [Saved config may become stale] → The design prefers the current explicit override and environment state over persisted values, but still allows the user to refresh the saved path by providing it again.
- [Some systems may have partial CCP4 installs] → The design verifies the required tools after setup and surfaces a clear error if they are still unavailable.
