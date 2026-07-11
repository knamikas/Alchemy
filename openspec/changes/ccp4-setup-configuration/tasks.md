## 1. Startup integration

- [x] 1.1 Add CCP4 readiness checks at the start of the main pipeline before any processing begins.
- [x] 1.2 Keep `--ccp4-setup` as an explicit override for advanced users without making it required for normal runs.

## 2. Discovery and persistence

- [x] 2.1 Resolve CCP4 setup in order: CLI override, PATH, `CCP4_SETUP`, saved JSON config, and common install locations.
- [x] 2.2 Prompt the user for a CCP4 setup path when CCP4 is unavailable and no saved path exists, then save it to JSON config.
- [x] 2.3 Reuse the saved value on later runs so the user does not need to repeat the setup path.

## 3. Error handling and validation

- [x] 3.1 Show a clear error explaining how to configure CCP4 correctly when it still cannot be resolved.
- [x] 3.2 Add tests for startup checks, saved-config reuse, prompt-and-save behavior, and override precedence.
