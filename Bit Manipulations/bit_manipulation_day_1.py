def convert_to_binary(n):
    """Convert an integer to its binary representation as a string."""
    if n == 0:
        return "0"
    binary_str = ""
    while n > 0:
        binary_str = str(n % 2) + binary_str
        n //= 2
    return binary_str

def convert_to_decimal(binary_str):
    """Convert a binary string to its decimal integer representation."""
    decimal_value = 0
    length = len(binary_str)
    for i in range(length):
        bit = int(binary_str[length - 1 - i])
        decimal_value += bit * (2 ** i)
    return decimal_value

def swap_two_numbers_xor(a, b):
    """Swap two numbers using XOR bitwise operation.""" 
    a = a ^ b
    print(f"After first XOR: a={a}, b={b}")
    b = a ^ b
    print(f"After second XOR: a={a}, b={b}")
    a = a ^ b 
    return a, b

def set_ith_bit(n, i):
    """Set the i-th bit of n to 1."""
    return n | (1 << i)

def clear_ith_bit(n, i):
    """Clear the i-th bit of n to 0."""
    return n & ~(1 << i)

def toggle_ith_bit(n, i):
    """Toggle the i-th bit of n."""
    return n ^ (1 << i)

def check_ith_bit(n, i):
    """Check if the i-th bit of n is set (1) or not (0)."""
    return (n & (1 << i)) != 0

def remove_last_set_bit(n):
    return n & (n - 1)

def sum_of_two_integers(a, b):
    """Sum two integers without using the '+' operator."""
    mask = 0xffffffff
    while mask & b:
        a, b = a ^ b, (a & b) << 1
    return (mask & a) if b > 0 else a

def divide_two_integers_with_abs(dividend, divisor):
    """
    Divide two integers without using multiplication, division, or mod operator.
    This function performs integer division using bit manipulation techniques.
    It works by finding the largest powers of 2 that fit into the dividend,
    subtracting them iteratively, and building the quotient using bitwise operations.
    Args:
        dividend (int): The number to be divided.
        divisor (int): The number to divide by.
    Returns:
        int: The quotient of dividend divided by divisor, rounded towards zero.
    Raises:
        ValueError: If divisor is zero, as division by zero is undefined.
    Algorithm:
        1. Check if divisor is zero and raise an error if true.
        2. Determine if the result should be negative based on signs of both operands.
        3. Convert both operands to absolute values for easier manipulation.
        4. Iterate through bit positions from 31 down to 0.
        5. For each bit position, check if the shifted divisor fits into the remainder.
        6. If it fits, subtract the shifted divisor and set the corresponding bit in quotient.
        7. Apply the negative sign if the operands had different signs.
    Time Complexity: O(log n) where n is the maximum value of dividend or divisor.
    Space Complexity: O(1) - only uses constant extra space.
    Example:
        >>> divide_two_integers(10, 3)
        3
        >>> divide_two_integers(-10, 3)
        -3
        >>> divide_two_integers(10, -3)
        -3
        Operation, Meaning,     Math Equivalent
        a << i,    Left shift,  a * 2**i
        a >> i,    Right shift, a / 2**i
        a ^= b,	   XOR,		    Determine if signs differ
        a |= b,	   Bitwise OR,  Set a specific bit to 1
    """ 
    MAX_INT = 2**31 - 1
    MIN_INT = -2**31

    if divisor == 0:
        raise ValueError("Divisor cannot be zero.")
    
    # Handle the specific overflow condition
    if dividend == MIN_INT and divisor == -1:
        return MAX_INT
    
    negative = (dividend < 0) ^ (divisor < 0)
    dividend, divisor = abs(dividend), abs(divisor)
    
    # Initialize the quotient to 0
    quotient = 0

    # Iterate through bit positions from 31 down to 0
    for i in reversed(range(32)):
        # At the start of each iteration, we check if the current bit position can fit
        # into the remaining dividend. We do this by shifting the divisor left by i bits.
        a = dividend >> i
        b = divisor << i
        if a >= divisor:
            # If it fits, we subtract the shifted divisor from the dividend
            dividend -= b
            # We also set the corresponding bit in the quotient
            quotient |= 1 << i
            
    ans = -quotient if negative else quotient
    return max(MIN_INT, min(ans, MAX_INT))


def divide_two_integers_without_abs(dividend: int, divisor: int) -> int:
    MAX_INT = 2**31 -1
    MIN_INT = -2**31

    if divisor == 0: raise ValueError("Number can not be divisible by 0!")

    if dividend == MIN_INT and divisor == -1:
        return MAX_INT
    
    negative = (dividend < 0) ^ (divisor < 0)
    quotient = 0

    if dividend > 0:
        dividend = -dividend
    if divisor > 0:
        divisor = -divisor

    for i in reversed(range(32)):
        b = divisor << i
        if b >= dividend and b < 0:
            dividend -= b
            quotient |= 1 << i
    ans = -quotient if negative else quotient
    return ans


if __name__ == "__main__":
    # Example usage
    num = 42
    binary_representation = convert_to_binary(num)
    print(f"Decimal: {num} -> Binary: {binary_representation}")

    binary_str = "101010"
    decimal_representation = convert_to_decimal(binary_str)
    print(f"Binary: {binary_str} -> Decimal: {decimal_representation}")
    a, b = 5, 6
    print(f"Original a: {a}, b: {b}")
    a, b = swap_two_numbers_xor(a, b)
    print(f"Swapped a: {a}, b: {b}")

    # print(divide_two_integers(-10, 0))
    print(sum_of_two_integers(2, 9))