from abc import ABC, abstractmethod
from typing import Protocol, Final


class EatsBread(Protocol):
    def eat_bread(self):
        pass


def feed_bread(animal: EatsBread):
    animal.eat_bread()


class Duck:
    def eat_bread(self):
        print("Quack!")
        print(self)


feed_bread(Duck())  # <-- OK


class Mees:
    def eat_bread(self):
        ...

    def drink_milk(self):
        ...


feed_bread(Mees())  # <-- OK


def foo(x: int = ...):
    if x is Ellipsis:
        print("x is ellipsis")
    if x is None:
        print("x is None")
    if x is ...:
        print("x is ...")
    return x


print(foo)
print("-" * 20)
print(foo())
print("-" * 20)
print(foo(None))
print("-" * 20)
print(foo(...))

from grablib.sorter import SorterBase


# a = SorterBase(1, 2)
# print(a)
# print(type(a))


class Demo(SorterBase):
    pass


from overrides import overrides, final

# from typing import final

a: Final = 20
print(a)
a = 30
print(a)


class A(ABC):
    # @final
    # @abstractmethod
    # def foo(self, heh):
    #     print('A.foo')

    pass


class B(A):
    @overrides(check_signature=False, check_at_runtime=False)
    def foo(self, hah):
        print("B.foo")

    # ...


B().foo(20)
