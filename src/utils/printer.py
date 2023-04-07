import sys
# enable colored prints
from termcolor import colored
import colorama
colorama.init()

# local import
from ..game import DotsAndBoxesGame


class Color:
    PLAYER_1 = "red"
    PLAYER_2 = "green"


class DotsAndBoxesPrinter(DotsAndBoxesGame):

    def __init__(self, size):
        super().__init__(size=size)

    """
    Add methods to the DotsAndBoxes game class the enable you to play the
    game in a console.
    """
    def state_string(self) -> str:
        boxes_player_1 = (self.b == 1).sum()
        boxes_player_2 = (self.b == -1).sum()
        s = "Score: (" + \
            colored("You", Color.PLAYER_1) + \
            f") {boxes_player_1}:{boxes_player_2} (" + \
            colored("Opponent", Color.PLAYER_2) + ")"
        return s


    def str_horizontal_line(self, line: int, last_column: bool) -> str:

        value = self.l[line]
        color = value_to_color(value)

        string = "°" + colored("------", color) if value != 0 else \
            "°  {: >2d}  ".format(line)
        return (string + "°") if last_column else string


    def str_vertical_line(self, left_line: int, print_line_number: bool) -> str:

        value = self.l[left_line]
        color = value_to_color(value)

        if value != 0:
            string = colored("|", color)

            # color the box when the box right to the line is already captured
            box = self.get_boxes_of_line(left_line)[-1]
            box_value = self.b[box[0], box[1]]
            if box_value == 0:
                return string + "      "
            else:
                color = value_to_color(box_value)
                return string + colored("======", color)

        else:
            if print_line_number:
                return "{: >2d}     ".format(left_line)
            else:
                return "       "


    def board_string(self) -> str:
        if self.SIZE > 6:
            sys.exit("ERROR: To ensure sufficient output quality in the console, the board size of games that are "
                     "printed is limited to 6.\n")

        # iterate through boxes from top to bottom, left to right
        string = ""
        for i in range(self.SIZE):

            # 1) use top line
            for j in range(self.SIZE):
                string += self.str_horizontal_line(
                    line=self.get_lines_of_box((i, j))[0],
                    last_column=(j == self.SIZE - 1)
                )
            string += "\n"

            # 2) use left and right lines
            for repeat in range(3):
                for j in range(self.SIZE):
                    string += self.str_vertical_line(
                        left_line=self.get_lines_of_box((i, j))[2],
                        print_line_number=(repeat == 1)
                    )

                # last vertical line in a row
                right_line = self.get_lines_of_box((i, self.SIZE - 1))[3]
                value = self.l[right_line]
                if value != 0:
                    string += colored("|", value_to_color(value))
                else:
                    if repeat == 1:
                        string += f"{right_line}"
                string += "\n"

            # 3) print bottom lines for the last row of boxes
            if i == self.SIZE - 1:
                for j in range(self.SIZE):
                    string += self.str_horizontal_line(
                        line=self.get_lines_of_box((i, j))[1],
                        last_column=(j == self.SIZE - 1)
                    )
                string += "\n"
        return string


def value_to_color(value: int):
    return Color.PLAYER_1 if value == 1 else Color.PLAYER_2
