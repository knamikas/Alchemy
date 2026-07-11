## Why

The main pipeline should handle CCP4 readiness itself rather than expecting users to remember a separate setup step every time. The goal is for normal runs to work when CCP4 is already on PATH, and to fail gracefully with clear guidance when it is not.

## What Changes

- Add CCP4 validation at the beginning of the main pipeline so the workflow checks whether `fft` and `edstats` are available before processing starts.
- If those tools are missing, try to resolve a CCP4 setup script from the current environment, saved configuration, and common install locations.
- If CCP4 is still not available, prompt the user to provide a CCP4 setup path once and save it in a JSON config file for future runs.
- Preserve the ability to override the detected or saved path with `--ccp4-setup` for advanced cases.
- If CCP4 cannot be found, exit with a clear error explaining how to configure it correctly.

## Capabilities

### New Capabilities
- `ccp4-startup-checks`: Validate CCP4 readiness before the main pipeline begins.
- `ccp4-setup-persistence`: Store a user-provided CCP4 setup path in JSON config for future use.
- `ccp4-setup-guidance`: Provide actionable setup guidance when CCP4 cannot be resolved.

### Modified Capabilities
- `pipeline-setup`: The main pipeline startup flow is changing to perform CCP4 validation and onboarding automatically.

## Impact

This change touches the entrypoint workflow in the main CLI, the shared CCP4 helper module, and the user-facing startup and error experience. It does not change the core analysis logic itself.
