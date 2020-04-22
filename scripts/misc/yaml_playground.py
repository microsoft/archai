import yaml

y = """
a: .NaN

"""

d=yaml.load(y, Loader=yaml.Loader)
print(d)
print(type( d['a']))
