class Matrix:

    def __init__(self, width: int, height: int, elements: list):
        if width * height != len(elements):
            raise Exception("The number of list items does not match the size of the matrix.")
        self.width = width
        self.height = height
        self.elements = elements
        self.matrix = []

        # empty matrix
        for _ in range(height):
            self.matrix.append([0] * width)
        # fill matrix
        for i in range(height):
            for j in range(width):
                self.matrix[i][j] = elements[(i * width) + j]

    def __str__(self):
        for row in self.matrix:
            row_str = ""
            for el in row:
                row_str += f"{el}\t"
            ### print(*row)
            print(row_str)
        # print("")
        return ""

    def __add__(self, other):
        if self.width != other.width or self.height != other.height:
            raise Exception("Matrix addition is not possible. The sizes of the matrices do not match.")

        zip_elements = zip(self.elements, other.elements)
        map_elements = map(sum, zip_elements)
        result = Matrix(width=self.width, height=self.height, elements=list(map_elements))
        return result

    def __sub__(self, other):
        if self.width != other.width or self.height != other.height:
            raise Exception("Matrix subtraction is not possible. The sizes of the matrices do not match.")

        other *= -1
        result = self + other
        return result

    def __mul__(self, other):
        # matrix multiplication by matrix
        if isinstance(other, Matrix):
            if self.width != other.height:
                raise Exception(f"Matrix multiplication is not possible. The sizes of the matrices do not match. "
                                f"self height = {self.width}, other width = {other.height}")

            new_matrix_width = other.width
            new_matrix_height = self.height
            new_matrix_elements = [0] * (self.height * other.width)
            new_matrix_common_size = other.height

            for i in range(0, new_matrix_height):
                for j in range(0, new_matrix_width):
                    for k in range(0, new_matrix_common_size):
                        try:
                            new_matrix_elements[new_matrix_width * i + j] += self.matrix[i][k] * other.matrix[k][j]
                        except Exception as ex:
                            print(ex)
            result = Matrix(width=new_matrix_width, height=new_matrix_height, elements=list(new_matrix_elements))
            return result

        # multiplying a matrix by a number
        if isinstance(other, int) or isinstance(other, float):
            new_matrix_elements = [element * other for element in self.elements]
            result = Matrix(width=self.width, height=self.height, elements=new_matrix_elements)
            return result

    def transpose(self):
        new_matrix_elements = [0] * (self.height * self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix_elements[(j * self.height) + i] = self.matrix[i][j]
        result = Matrix(width=self.height, height=self.width, elements=new_matrix_elements)
        return result

    def getList(self):
        values = []
        for row in self.matrix:
            for value in row:
                values.append(value)
        self.elements = values
        return values
