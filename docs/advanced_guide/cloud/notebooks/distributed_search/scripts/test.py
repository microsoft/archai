def foo():
    def bar(x):
        print(x)


    bar(10)
    bar("xyz")

foo()