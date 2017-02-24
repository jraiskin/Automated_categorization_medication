import argparse
# https://pymotw.com/2/argparse/

parser = argparse.ArgumentParser()

parser.add_argument('-s', action='store', dest='simple_string',
                    help='Store a simple string', type=str,
                    default='no default given')

parser.add_argument('-a', action='store', dest='value_a',
                    help='int 1', type=int,
                    default=42)

parser.add_argument('-b', action='store', dest='value_b',
                    help='int 2', type=int)

results = parser.parse_args()

print(results)
print('simple_string =', results.simple_string)
print('value_a =', results.value_a)
print('value_b =', results.value_b)
print('value_b =', value_b)
