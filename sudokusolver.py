from itertools import islice
from typing import Iterable, Union, Tuple

import numpy as np


def solve(field: Union[str, np.ndarray, list, tuple], cell_shape: Tuple[int, int] = None) -> Iterable[np.ndarray]:
    """
    Solves a sudoku puzzle and yields all possible solutions.
    Parameters
    ----------
    field
        if array - nan values denote unfilled positions, if str - '?' denotes unfilled positions
    cell_shape
        the shape of a single cell (e.g. simple sudoku = (3, 3) ), supports non-square cells.
        if None - the shape is considered square and is inferred from field's shape
    """
    if isinstance(field, str):
        field, initial, cell_shape = parse_text(field, cell_shape)
    else:
        field, initial, cell_shape = parse_array(field, cell_shape)

    for v in initial:
        propagate_constraints(field, *v, cell_shape)
    return make_suggestions(field, cell_shape)


class NotSolvable(ValueError):
    pass


def propagate_constraints(field, i, j, value, cell_shape):
    mask = field[..., value]
    update_at = np.zeros_like(mask)

    # rows, cols
    update_at[i] = 1
    update_at[:, j] = 1

    # cell
    start = (np.array((i, j)) // cell_shape) * cell_shape
    stop = start + cell_shape
    update_at[tuple(map(slice, start, stop))] = 1

    # except
    update_at[i, j] = 0

    # only where should remove
    update_at = update_at & mask
    mask[update_at] = 0

    if not field.any(-1).all():
        # no possible values
        raise NotSolvable

    # propagate
    update_at = update_at & (field.sum(-1) == 1)
    xs, ys = np.where(update_at)
    for i, j in zip(xs, ys):
        unique_value, = np.where(field[i, j])[0]
        propagate_constraints(field, i, j, unique_value, cell_shape)


def is_solved(field):
    return (field.sum(-1) == 1).all()


def make_suggestions(field, cell_shape):
    if is_solved(field):
        values = np.where(field)[2] + 1
        yield values.reshape(*field.shape[:2])
        return

    variants = field.sum(-1).astype(float)
    variants[variants == 1] = np.inf
    idx = np.argmin(variants)
    i, j = np.unravel_index(idx, field.shape[:2])

    for value in range(len(field)):
        if field[i, j, value]:
            suggestion = field.copy()
            # suppose that the real value at (i,j) is `value`
            suggestion[i, j] = 0
            suggestion[i, j, value] = 1

            try:
                propagate_constraints(suggestion, i, j, value, cell_shape)
                yield from make_suggestions(suggestion, cell_shape)
            except NotSolvable:
                pass


def get_cell_shape(field_side, cell_shape):
    if cell_shape is None:
        cell_side = int(np.sqrt(field_side))
        assert cell_side ** 2 == field_side
        cell_shape = np.array((cell_side, cell_side), int)

    cell_size = np.prod(cell_shape)
    assert field_side == cell_size
    return cell_shape, cell_size


def parse_text(text: str, cell_shape):
    values = text.split()
    field_side = int(np.sqrt(len(values)))
    if len(values) != field_side ** 2:
        raise ValueError('The field is not square.')

    cell_shape, cell_size = get_cell_shape(field_side, cell_shape)
    field = np.ones([field_side] * 3, dtype=bool)

    initial = []
    for i, value in enumerate(values):
        if value != '?':
            value = int(value)
            assert 0 < value <= cell_size

            i, j = np.unravel_index(i, field.shape[:2])
            value -= 1
            field[i, j] = 0
            field[i, j, value] = 1
            initial.append((i, j, value))

    return field, initial, cell_shape


def parse_array(array: np.ndarray, cell_shape):
    array = np.asarray(array, float)
    field_side = len(array)
    if array.ndim != 2 or field_side != array.shape[1]:
        raise ValueError('The field is not square.')

    cell_shape, cell_size = get_cell_shape(field_side, cell_shape)
    field = np.ones([field_side] * 3, dtype=bool)
    not_empty = ~np.isnan(array)

    initial = []
    for i, j in zip(*not_empty.nonzero()):
        value = int(array[i, j])
        assert 0 < value <= cell_size

        value -= 1
        field[i, j] = 0
        field[i, j, value] = 1
        initial.append((i, j, value))

    return field, initial, cell_shape


def print_field(field, cell_shape):
    cell_shape, cell_size = get_cell_shape(len(field), cell_shape)
    h, w = cell_shape
    number_width = int(np.log10(cell_size - 1)) + 1
    result_width = cell_size * (number_width + 1) + 2 * cell_size // w + 1

    for i, row in enumerate(field):
        if i % h == 0:
            print('-' * result_width)

        for j, value in enumerate(row):
            value = str(value).ljust(number_width + 1, ' ')

            if j % w == 0:
                print('| ', end='')
            print(value, end='')
        print('|')
    print('-' * result_width)


def print_solutions(text: str, cell_shape: Tuple[int, int] = None, max_solutions: int = None):
    solutions = solve(text, cell_shape)
    if max_solutions is not None:
        solutions = islice(solutions, max_solutions)

    for solution in solutions:
        print_field(solution, cell_shape)
        # for row in solution:
        #     print(*row)
        print()