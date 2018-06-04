import configparser


class Parameter:
    FIELD_NAME = "parameter"

    def __init__(self, filename, check=False):
        self._parser = configparser.ConfigParser()
        self._parser.read(filename)
        self._params = {}
        self.init_parameters()
        if check:
            self.print()

    def init_parameters(self):
        for param in self._parser[Parameter.FIELD_NAME]:
            value = self._parser[Parameter.FIELD_NAME][param]
            self._params[param] = Parameter.parse_to_variable(value)

    def get(self, index):
        try:
            return self._params[index]
        except KeyError:
            raise ValueError("Parameter object doesn't have that index.")

    def print(self):
        print("[Parameters]")
        for param in self._params:
            print(param, self._params[param])
        print("")

    @staticmethod
    def parse_to_variable(var_str):
        if var_str == "True":
            return True
        elif var_str == "False":
            return False
        elif var_str == "None":
            return None
        elif "." in var_str:
            return float(var_str)
        else:
            return int(var_str)


if __name__ == "__main__":
    p = Parameter("../parameter.ini")
    p.print()
