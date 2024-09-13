"""
User provide initial known values in the sudoku.

The program solve the sudoku.

Sudoku rules: The numbers 1,2,3,4,5,6,7,8,9 should all be present in each row, column and (3x3)-superblock

Solver idea:
- Iteratively solve the puzzle.
- Use the sudoku rules to restrict the possible numbers at each block.
  One or many blocks can often be determined from this.
- If no block can be determined, make a guess in one block.
  Also keep track of the guess history and the block values that can
  be inferred from this guess. 
- If a guess is wrong it will lead to a inconsistency in the puzzle.
  Then revert wrong guesses and make a new guess.  
- The puzzle is solved when all blocks have a number. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


def plot_sudoku_board(
    matrix, possible_numbers=None, guesses=None, inferred_values=None
):
    assert matrix.shape == (9, 9)

    fig, ax = plt.subplots()

    for i in range(9):
        for j in range(9):
            # 0 matrix elements is not shown
            text = f"{matrix[i,j]}" if matrix[i, j] else " "

            rect = patches.Rectangle(
                (j, 8 - i), 1, 1, linewidth=1, edgecolor="grey", facecolor="lightgrey"
            )
            ax.add_patch(rect)
            plt.text(j + 0.4, 8 - i + 0.3, text, fontsize=14)

            if not matrix[i, j] and possible_numbers is not None:
                values = possible_numbers[i, j, :]
                values = values[np.where(values)]
                for value in values:
                    dx = 0.25 * (((value - 1) % 3) - 1)
                    dy = -0.25 * ((value - 1) // 3 - 1)
                    plt.text(j + 0.4 + dx, 8 - i + 0.3 + dy, f"{value}", fontsize=8)

    if guesses is not None:
        for guess in guesses:
            (i, j), p, k = guess
            plt.text(j + 0.4, 8 - i + 0.3, f"{p[k]}", fontsize=14, color="red")
    if inferred_values is not None:
        for inferred_matrix in inferred_values:
            for i, j in zip(*np.where(inferred_matrix)):
                plt.text(
                    j + 0.4,
                    8 - i + 0.3,
                    f"{inferred_matrix[i,j]}",
                    fontsize=14,
                    color="orange",
                )

    # Draw 9 thick rectangles
    for i in range(3):
        for j in range(3):
            rect = patches.Rectangle(
                (3 * j, 3 * (2 - i)),
                3,
                3,
                linewidth=2,
                edgecolor="black",
                facecolor="none",
            )
            ax.add_patch(rect)

    plt.xlim([-0.1, 9.1])
    plt.ylim([-0.1, 9.1])
    plt.axis("off")
    plt.title("Sudoku")
    plt.show()


def get_initial_matrix(mode: str):
    if mode == "easy1":
        initial_matrix = np.zeros((9, 9), dtype=int)
        initial_matrix[0, [3, 4, 6, 8]] = [2, 6, 7, 1]
        initial_matrix[1, [0, 1, 4, 7]] = [6, 8, 7, 9]
        initial_matrix[2, [0, 1, 5, 6]] = [1, 9, 4, 5]
        initial_matrix[3, [0, 1, 3, 7]] = [8, 2, 1, 4]
        initial_matrix[4, [2, 3, 5, 6]] = [4, 6, 2, 9]
        initial_matrix[5, [1, 5, 7, 8]] = [5, 3, 2, 8]
        initial_matrix[6, [2, 3, 7, 8]] = [9, 3, 7, 4]
        initial_matrix[7, [1, 4, 7, 8]] = [4, 5, 3, 6]
        initial_matrix[8, [0, 2, 4, 5]] = [7, 3, 1, 8]
    elif mode == "easy2":
        initial_matrix = np.zeros((9, 9), dtype=int)
        initial_matrix[0, [0, 3, 4, 5, 8]] = [1, 4, 8, 9, 6]
        initial_matrix[1, [0, 1, 7]] = [7, 3, 4]
        initial_matrix[2, [5, 6, 7, 8]] = [1, 2, 9, 5]
        initial_matrix[3, [2, 3, 4, 6]] = [7, 1, 2, 6]
        initial_matrix[4, [0, 3, 5, 8]] = [5, 7, 3, 8]
        initial_matrix[5, [2, 4, 5, 6]] = [6, 9, 5, 7]
        initial_matrix[6, [0, 1, 2, 3]] = [9, 1, 4, 6]
        initial_matrix[7, [1, 7, 8]] = [2, 3, 7]
        initial_matrix[8, [0, 3, 4, 5, 8]] = [8, 5, 1, 2, 4]
    elif mode == "intermediate1":
        initial_matrix = np.zeros((9, 9), dtype=int)
        initial_matrix[0, [1, 3, 5]] = [2, 6, 8]
        initial_matrix[1, [0, 1, 5, 6]] = [5, 8, 9, 7]
        initial_matrix[2, [4]] = [4]
        initial_matrix[3, [0, 1, 6]] = [3, 7, 5]
        initial_matrix[4, [0, 8]] = [6, 4]
        initial_matrix[5, [2, 7, 8]] = [8, 1, 3]
        initial_matrix[6, [4]] = [2]
        initial_matrix[7, [2, 3, 7, 8]] = [9, 8, 3, 6]
        initial_matrix[8, [3, 5, 7]] = [3, 6, 9]
    elif mode == "difficult1":
        initial_matrix = np.zeros((9, 9), dtype=int)
        initial_matrix[0, [3, 6]] = [6, 4]
        initial_matrix[1, [0, 5, 6]] = [7, 3, 6]
        initial_matrix[2, [4, 5, 7]] = [9, 1, 8]
        initial_matrix[4, [1, 3, 4, 8]] = [5, 1, 8, 3]
        initial_matrix[5, [3, 5, 7, 8]] = [3, 6, 4, 5]
        initial_matrix[6, [1, 3, 7]] = [4, 2, 6]
        initial_matrix[7, [0, 2]] = [9, 3]
        initial_matrix[8, [1, 6]] = [2, 1]
    elif mode == "difficult2":
        initial_matrix = np.zeros((9, 9), dtype=int)
        initial_matrix[0, [0, 3]] = [2, 3]
        initial_matrix[1, [0, 2, 4, 5, 8]] = [8, 4, 6, 2, 3]
        initial_matrix[2, [1, 2, 3, 6]] = [1, 3, 8, 2]
        initial_matrix[3, [4, 6, 7]] = [2, 3, 9]
        initial_matrix[4, [0, 2, 6, 7, 8]] = [5, 7, 6, 2, 1]
        initial_matrix[5, [1, 2, 5]] = [3, 2, 6]
        initial_matrix[6, [1, 5, 6, 7]] = [2, 9, 1, 4]
        initial_matrix[7, [0, 2, 3, 4, 6, 8]] = [6, 1, 2, 5, 8, 9]
        initial_matrix[8, [5, 8]] = [1, 2]
    elif mode == "extreme1":
        initial_matrix = np.zeros((9, 9), dtype=int)
        initial_matrix[0, [1]] = [2]
        initial_matrix[1, [3, 8]] = [6, 3]
        initial_matrix[2, [1, 2, 4]] = [7, 4, 8]
        initial_matrix[3, [5, 8]] = [3, 2]
        initial_matrix[4, [1, 4, 7]] = [8, 4, 1]
        initial_matrix[5, [0, 3]] = [6, 5]
        initial_matrix[6, [4, 6, 7]] = [1, 7, 8]
        initial_matrix[7, [0, 5]] = [5, 9]
        initial_matrix[8, [7]] = [4]
    else:
        n = int(input("Enter the number of (initially) known blocks: "))
        assert 0 < n < 9**2

        initial_matrix = np.zeros((9, 9), dtype=int)
        counter = 0
        print("Please enter the row, column, and the value of each such block.")
        while True:
            print(
                f"Please enter information about block {counter+1} (out of {n} blocks)"
            )
            i = int(input("Row (1-9): "))
            if i < 1 or i > 9:
                print(f"Wrong row input: {i}, please fill in the block again")
                continue
            j = int(input("Column (1-9): "))
            if j < 1 or j > 9:
                print(f"Wrong column input: {j}, please fill in the block again")
                continue
            if initial_matrix[i - 1, j - 1] != 0:
                print(
                    f"Block with row {i} and column {j} is already filled in, please fill in the block again"
                )
                continue
            v = int(input("Value (1-9): "))
            if v < 1 or v > 9:
                print(f"Wrong block value input: {v}, please fill in the block again")
                continue

            assert 1 <= i <= 9
            assert 1 <= j <= 9
            assert 1 <= v <= 9
            assert initial_matrix[i - 1, j - 1] == 0
            # 0-indexing instead of 1-indexing
            initial_matrix[i - 1, j - 1] = v
            counter += 1
            if counter == n:
                break
        print("Succesfully entered all blocks!")

    assert_matrix(initial_matrix)
    return initial_matrix


def assert_matrix(matrix):
    assert matrix.dtype == int
    assert matrix.shape == (9, 9)
    # Assert 0 or values 1,2,3,4,5,6,7,8,9
    assert np.all(np.logical_and(0 <= matrix, matrix <= 9))
    # Assert row rule
    for i in range(9):
        row = matrix[i, :]
        knowns = row[np.where(row)]
        # Assert unique values
        len(knowns) == len(set(knowns))
    # Assert column rule
    for j in range(9):
        column = matrix[:, j]
        knowns = column[np.where(column)]
        # Assert unique values
        len(knowns) == len(set(knowns))
    # Assert superblock rule
    for i in range(3):
        for j in range(3):
            superblock = matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
            knowns = superblock[np.where(superblock)]
            # Assert unique values
            len(knowns) == len(set(knowns))


def get_possible_numbers(matrix):
    assert_matrix(matrix)
    possible_numbers = np.zeros((9, 9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            if matrix[i, j] > 0:
                # Already set, do nothing
                continue
            superblock_i = i // 3
            superblock_j = j // 3

            # Take the union of the already taken values
            row = matrix[i, :]
            column = matrix[:, j]
            superblock = matrix[
                3 * superblock_i : 3 * superblock_i + 3,
                3 * superblock_j : 3 * superblock_j + 3,
            ]
            taken_values = (
                set(tuple(row[np.where(row)]))
                | set(tuple(column[np.where(column)]))
                | set(tuple(superblock[np.where(superblock)]))
            )
            p = set((1, 2, 3, 4, 5, 6, 7, 8, 9)) - taken_values
            if len(p) == 0:
                raise InconsistentMatrix
            possible_numbers[i, j, : len(p)] = tuple(p)
    return possible_numbers


def get_new_blocks(matrix, possible_numbers):
    # Loop over unknown blocks, check if only on number is possible
    i_unknowns, j_unknowns = np.where(matrix == 0)
    new_blocks1 = np.zeros((9, 9), dtype=int)
    for i, j in zip(i_unknowns, j_unknowns):
        p = possible_numbers[i, j, :]
        p = p[np.where(p)]
        if len(p) == 1:
            # print(f"{p[0]} should be at row {i+1} and column {j+1}")
            new_blocks1[i, j] = p[0]

    # For each rule, loop over 1,2,...9 and for each number check how many blocks can have this number.
    # If only one block -> the number belong to that block!

    # row rule
    new_blocks2 = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for v in range(1, 10):
            row = matrix[i, :]
            if v in row:
                continue
            mask = np.sum(possible_numbers[i, :, :] == v, axis=-1)
            np.testing.assert_equal(np.logical_or(mask == 0, mask == 1), 1)
            if np.sum(mask) == 0:
                raise InconsistentMatrix
            if np.sum(mask) == 1:
                # Only one block can have this number
                j = np.where(mask)[0][0]
                if new_blocks2[i, j] != 0:
                    raise InconsistentMatrix
                new_blocks2[i, j] = v

    # column rule
    new_blocks3 = np.zeros((9, 9), dtype=int)
    for j in range(9):
        for v in range(1, 10):
            column = matrix[:, j]
            if v in column:
                continue
            mask = np.sum(possible_numbers[:, j, :] == v, axis=-1)
            np.testing.assert_equal(np.logical_or(mask == 0, mask == 1), 1)
            if np.sum(mask) == 0:
                raise InconsistentMatrix
            if np.sum(mask) == 1:
                # Only one block can have this number
                i = np.where(mask)[0][0]
                if new_blocks3[i, j] != 0:
                    raise InconsistentMatrix
                new_blocks3[i, j] = v

    # superblock rule
    new_blocks4 = np.zeros((9, 9), dtype=int)
    for superblock_i in range(3):
        for superblock_j in range(3):
            for v in range(1, 10):
                superblock = matrix[
                    3 * superblock_i : 3 * superblock_i + 3,
                    3 * superblock_j : 3 * superblock_j + 3,
                ]
                if v in superblock:
                    continue
                mask = np.sum(
                    possible_numbers[
                        3 * superblock_i : 3 * superblock_i + 3,
                        3 * superblock_j : 3 * superblock_j + 3,
                        :,
                    ]
                    == v,
                    axis=-1,
                )
                np.testing.assert_equal(np.logical_or(mask == 0, mask == 1), 1)
                if np.sum(mask) == 0:
                    raise InconsistentMatrix
                if np.sum(mask) == 1:
                    # Only one block can have this number
                    i, j = np.where(mask)
                    i = i[0]
                    j = j[0]
                    i += 3 * superblock_i
                    j += 3 * superblock_j
                    if new_blocks4[i, j] != 0:
                        raise InconsistentMatrix
                    new_blocks4[i, j] = v

    new_blocks = np.array([new_blocks1, new_blocks2, new_blocks3, new_blocks4])
    new_blocks = np.max(new_blocks, axis=0)
    return new_blocks


class InconsistentMatrix(Exception):
    # Raised when a sudoku matrix is inconsistent, probably due to a wrong guess.
    pass


def revert_bad_guesses_and_make_a_new_guess(blocks, guesses, inferred_values):
    """In-place modify input arguments."""
    (i, j), p, k = guesses[-1]
    if 0 <= k < len(p) - 1:
        # Revert inferred block values from bad guess
        blocks[inferred_values[-1] > 0] = 0
        inferred_values[-1][...] = 0

        # Make a new guess at the same block: next possible value
        guesses[-1] = ((i, j), p, k + 1)
        blocks[i, j] = p[k + 1]

        print(
            f"At row {i+1} and column {j+1}, revert wrong guess {p[k]} and try {p[k+1]} instead"
        )
    else:
        # At least the last two guesses were wrong.
        # First clean up from the last wrong guess
        print("Multiple guesses are wrong and will be fixed!")
        print(f"At row {i+1} and column {j+1}, revert wrong guess {p[k]}.")
        blocks[i, j] = 0
        blocks[inferred_values[-1] > 0] = 0
        guesses.pop(-1)
        inferred_values.pop(-1)
        # Now we have to fix the second last wrong guess,
        # and possible also even earlier guesses.
        revert_bad_guesses_and_make_a_new_guess(blocks, guesses, inferred_values)


def solve_sudoku(initial_matrix, plot_intermediate_results=False):
    assert_matrix(initial_matrix)

    # We might have to make guesses.
    # Store these guesses in this variable.
    guesses = []
    # After a guess we hopefully can infer new values.
    # Store these in this variable.
    inferred_values = []

    # Contain known + guessed + inferred blocks
    blocks = initial_matrix.copy()

    # Iteratively fill in blocks untill all blocks are known.
    # For hard problems the algorithm need to guess values in
    # order to be able to continue.
    # If the guess was wrong it will lead to an inconsistency.
    # Then revert wrong guesses and make a new guess.
    while True:
        unknowns = blocks == 0
        print(f"Number of unknowns: {np.sum(unknowns)}")
        if np.sum(unknowns) == 0:
            # Puzzle is solved!
            break

        try:
            # For each block calculate the possible numbers that simply fulfill the row, column, and superblock rule
            possible_numbers = get_possible_numbers(blocks)
            new_blocks = get_new_blocks(blocks, possible_numbers)
        except InconsistentMatrix:
            revert_bad_guesses_and_make_a_new_guess(blocks, guesses, inferred_values)
            continue

        print(f"Found {np.sum(new_blocks > 0)} new blocks")

        if plot_intermediate_results:
            plot_sudoku_board(blocks, possible_numbers, guesses, inferred_values)

        if np.sum(new_blocks) == 0:
            # No new block determined.
            # Make one guess in order to be able to continue.

            # Guess in a block with fewest number of possible numbers
            i_guess, j_guess, n_possible = 10, 10, 10
            for i, j in zip(*np.where(unknowns)):
                p = possible_numbers[i, j, :]
                if len(p[np.where(p)]) < n_possible:
                    i_guess, j_guess, n_possible = i, j, len(p[np.where(p)])
            assert 1 < n_possible < 10
            p = possible_numbers[i_guess, j_guess, :]
            p = p[np.where(p)]

            guesses.append(((i_guess, j_guess), tuple(p), 0))
            assert blocks[i_guess, j_guess] == 0
            blocks[i_guess, j_guess] = p[0]
            print(f"Guess {p[0]} at row {i_guess+1} and column {j_guess+1}")
            inferred_values.append(np.zeros((9, 9), dtype=int))
        else:
            if len(inferred_values) > 0:
                inferred_values[-1] += new_blocks
            blocks += new_blocks

        assert_matrix(blocks)
    assert_matrix(blocks)
    # No unknown block left
    assert np.sum(blocks == 0) == 0
    return blocks


def main():
    initial_matrix = get_initial_matrix(mode="easy1")
    plot_sudoku_board(initial_matrix)

    t = time.time()
    solution_matrix = solve_sudoku(initial_matrix, plot_intermediate_results=False)
    t = time.time() - t
    print(f"Took {t :.1f} seconds to solve the sudoku")

    plot_sudoku_board(solution_matrix)


if __name__ == "__main__":
    main()
