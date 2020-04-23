import sys
import tester


def main(config_paths):
    for config_path in config_paths:
        with open(config_path, 'r') as file:
            json_config = file.read()
            config = tester.TestConfig.from_json(json_config)
        print(f"Test {config.test_name} started")
        tester.perform_test(config)


if __name__ == '__main__':
    main(sys.argv[1:])
