import sys
import tester


def main(config_path):
    with open(config_path, 'r') as file:
        json_config = file.read()
        config = tester.TestConfig.from_json(json_config)
    _ = tester.perform_single_test(config)


if __name__ == '__main__':
    main(sys.argv[1])
