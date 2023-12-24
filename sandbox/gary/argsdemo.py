from attrs import define, field
from dataclasses import InitVar
from datargs import make_parser


@define
class A:
    x: int
    y: int = 0


@define
class B:
    z: int = field(init=False)
    # x: int = 0

    ha: int = field(metadata={"help": "haha"})

    def haha(self):
        self.ha = 0


if __name__ == "__main__":
    parser = make_parser(A)
    parser = make_parser(B, parser=parser)
    args = parser.parse_args()

    print(args)
