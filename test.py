# import argparse

# if __name__ == "__main__":
#     arg_parser = argparse.ArgumentParser()
#     arg_parser.add_argument("--name", "-n",  default="lycon", type=str)
#     arg_parser.add_argument("--age", "-a", default=25, type=int)  # we can parse as many arguments with command prompt as well
#     # what is this --name? see when we write python --version, it displays python version. --name is same as that, instead of writing full --name we can write -n as well for the same
#     parse_args = arg_parser.parse_args()

#     print(parse_args.name, parse_args.age)
#     # if we don't give and value to arguments we'll be getting the default values
#     # to run it just write, py test.py --name "Raj" --age 24
#     # the output is going to be -> Raj 24

#     # this is an alternative to test on parameters instead of implicitly deploying Search CVs

import mlflow

dictionary = {
    'a' : [1,2,3],
    'b' : [4,5,6]
}

print()