# %%
import numpy as np
import pandas as pd
import numbers
from fractions import Fraction
import timeit


def compare_isscalar_isinstance():
    # Define a range of input values to test
    inputs = [
        42,  # integer
        3.14,  # float
        1 + 2j,  # complex number
        Fraction(1, 3),  # fraction
        "text",  # string
        [1, 2, 3],  # list
        (4, 5, 6),  # tuple
        {7, 8, 9},  # set
        {"a": 10},  # dictionary
        np.array([1, 2, 3]),  # numpy array
        np.array(10),  # single-element numpy array (scalar-like)
        np.float32(2.5),  # numpy float32
        np.int64(7),  # numpy int64
        np.complex128(3 + 4j),  # numpy complex128
        None,  # NoneType
        True,  # boolean
        False,  # boolean
        b"bytes",  # bytes
        bytearray(b"bytes"),  # bytearray
    ]

    # Create a list to store results
    results = []

    # Iterate over each input and check both conditions
    for item in inputs:
        # Measure the time for np.isscalar
        is_scalar_time = timeit.timeit(lambda: np.isscalar(item), number=1000)
        is_scalar = np.isscalar(item)

        # Measure the time for isinstance check
        is_number_time = timeit.timeit(
            lambda: isinstance(item, numbers.Number), number=1000
        )
        is_number = isinstance(item, numbers.Number)

        # Compare times
        if is_scalar_time < is_number_time:
            faster_approach = "np.isscalar"
        elif is_scalar_time > is_number_time:
            faster_approach = "isinstance"
        else:
            faster_approach = "Equal"

        same_result = is_scalar == is_number
        results.append(
            {
                "input": repr(item),
                "np.isscalar": is_scalar,
                "isinstance(x, numbers.Number)": is_number,
                "same_result": same_result,
                "np.isscalar_time": is_scalar_time,
                "isinstance_time": is_number_time,
                "faster_approach": faster_approach,
            }
        )

    # Convert the results to a DataFrame
    df = pd.DataFrame(results)

    # Define a function for coloring cells based on True/False values
    def color_code(val):
        color = "lightgreen" if val else "lightcoral"
        return f"background-color: {color}"

    # Define a function for color coding the faster_approach column
    def color_faster(val):
        if val == "np.isscalar":
            return "background-color: lightblue"
        elif val == "isinstance":
            return "background-color: lightyellow"
        else:  # "Equal"
            return "background-color: lightgray"

    # Apply color coding to the boolean columns and the faster_approach column
    styled_df = df.style.applymap(
        color_code,
        subset=["np.isscalar", "isinstance(x, numbers.Number)", "same_result"],
    )
    styled_df = styled_df.applymap(color_faster, subset=["faster_approach"])

    return styled_df


# Run the function and display the styled DataFrame
compare_isscalar_isinstance()
