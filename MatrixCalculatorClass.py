class MatrixCalculator:

    class Matrix:
        def __init__(self):
            self.width = 0
            self.height = 0
            self.matrix = []
            self.elements = []

        def create(self, elements: list, width: int, height: int):
            if height * width == len(elements):
                # save data
                self.width = width
                self.height = height
                self.elements = elements
                # make empty matrix
                for _ in range(self.height):
                    self.matrix.append([0] * self.width)
                # fill matrix
                for i in range(self.height):
                    for j in range(self.width):
                        self.matrix[i][j] = elements[i * self.width + j]
                return self
            else:
                raise Exception("The number of list items does not match the size of the matrix")

        def print(self):
            for row in self.matrix:
                print(*row)

        def __str__(self):
            self.print()
            return ""

    def sum(self, matrix_1: Matrix, matrix_2: Matrix):
        if matrix_1.width == matrix_2.width and matrix_1.height == matrix_2.height:
            zip_elements = zip(matrix_1.elements, matrix_2.elements)
            map_elements = map(sum, zip_elements)
            result = self.Matrix().create(list(map_elements), matrix_1.width, matrix_1.height)
            return result
        else:
            raise Exception("Matrix addition is not possible. The sizes of the matrices do not match.")

    def diff(self, matrix_1: Matrix, matrix_2: Matrix):
        if matrix_1.width == matrix_2.width and matrix_1.height == matrix_2.height:
            matrix_2 = self.multipleNum(matrix_2, -1)
            result = self.sum(matrix_1, matrix_2)
            return result
        else:
            raise Exception("Matrix subtraction is not possible. The sizes of the matrices do not match.")

    def multiple(self, matrix_1: Matrix, matrix_2: Matrix):
        if matrix_1.width == matrix_2.height:
            matrix_3_elements = [0] * (matrix_1.height * matrix_2.width)
            matrix_3_width = matrix_2.width
            matrix_3_height = matrix_1.height
            matrix_3_common_size = matrix_2.height
            for i in range(0, matrix_3_height):
                for j in range(0, matrix_3_width):
                    for k in range(0, matrix_3_common_size):
                        try:
                            matrix_3_elements[matrix_3_width * i + j] += matrix_1.matrix[i][k] * matrix_2.matrix[k][j]
                        except Exception as _:
                            pass
            result = self.Matrix().create(list(matrix_3_elements), width=matrix_3_width, height=matrix_3_height)
            return result
        else:
            raise Exception(f"Matrix multiplication is not possible. The sizes of the matrices do not match. "
                            f"matrix_1 height = {matrix_1.width}, matrix_2 width = {matrix_2.height}")

    def multipleNum(self, matrix: Matrix, num):
        new_matrix = [x * num for x in matrix.elements]
        result = self.Matrix().create(new_matrix, matrix.width, matrix.height)
        return result

    def transpose(self, matrix: Matrix):
        new_matrix = [0] * (matrix.height * matrix.width)
        for i in range(matrix.height):
            for j in range(matrix.width):
                new_matrix[j * matrix.height + i] = matrix.matrix[i][j]
        result = self.Matrix().create(new_matrix, matrix.height, matrix.width)
        return result

    def getListFromMatrix(self, matrix: Matrix):
        values = []
        for row in matrix.matrix:
            for value in row:
                values.append(value)
        return values

    def average(self, matrix: Matrix):
        all_sum = 0
        for _ in matrix.matrix:
            for value in matrix.matrix:
                all_sum += value
        return all_sum / (matrix.width * matrix.height)


##### TESTS #####
"""
calc = MatrixCalculator()
matrix_1 = calc.Matrix().create([9, 3, 5, 2, 0, 3, 0, 1, -6], 3, 3)
matrix_2 = calc.Matrix().create([1, -1, -1, -1, 4, 7, 8, 1, -1], 3, 3)
matrix_3 = calc.Matrix().create([1, 2, 3, 4, 5, 6], 2, 3)
matrix_1.print()
print("")
matrix_2.print()
print("")
calc.sum(matrix_1, matrix_2).print()
print("")
calc.diff(matrix_1, matrix_2).print()
print("")
calc.multiple(matrix_1, matrix_2).print()
print("")
calc.transpose(matrix_3).print()
print("")
calc.multipleNum(matrix_3, 3.2).print()
print("")
calc = MatrixCalculator()
matrix_4 = calc.Matrix().create([1, 2, 3, 3, 2, 1], 3, 2)
matrix_5 = calc.Matrix().create([1, -1, -1, 1], 2, 2)
matrix_4.print()
print("")
matrix_5.print()
print("")
calc.multiple(matrix_5, matrix_4).print()
print("")
"""