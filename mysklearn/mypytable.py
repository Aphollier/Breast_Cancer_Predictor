from mysklearn import classifier_utils
import copy
import csv
from tabulate import tabulate 
# uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """

        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """

        try:
            if col_identifier in self.column_names:
                pass
            else:
                raise ValueError
        except ValueError:
            print(col_identifier, "not a column in the table")
        col = []
        c_index = self.column_names.index(col_identifier)
        for row in self.data:
            if include_missing_values:
                col.append(row[c_index])
            else:
                if row[c_index] != "NA":
                    col.append(row[c_index])
        return col

    def rem_column(self, col_identifier):
        """Remove a column from the data

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index

        Notes:
            Raise ValueError on invalid col_identifier
        """
        try:
            if col_identifier in self.column_names:
                pass
            else:
                raise ValueError
        except ValueError:
            print(col_identifier, "not a column in the table")
        ind = self.column_names.index(col_identifier)
        self.column_names.remove(col_identifier)
        for row in self.data:
            del row[ind]

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for element in row:
                try:
                    row[row.index(element)] = float(element)
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        for row_index in sorted(row_indexes_to_drop, reverse=True):
            self.data.pop(row_index)


    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        self.data.clear()
        self.column_names.clear()
        infile = open(filename, "r")
        reader = csv.reader(infile)
        for row in reader:
            if self.column_names == []:
                self.column_names = row
            else:
                self.data.append(row)
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")
        writer = csv.writer(outfile)
        writer.writerow(self.column_names)
        writer.writerows(self.data)
        outfile.close

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        cols = []
        indexes = []
        no_dupes = []
        for key in key_column_names:
            cols.append(self.get_column(key))
        for i in range(len(cols[0])):
            composite = ""
            for j in range(len(cols)):
                composite += str(cols[j][i])
            if composite in no_dupes:
                indexes.append(len(no_dupes))
            no_dupes.append(composite)

        return indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        for row in reversed(self.data):
            if 'NA' in row:
                self.data.remove(row)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col = self.get_column(col_name, False)
        col_avg = sum(col)/len(col)
        index = self.column_names.index(col_name)
        for row in self.data:
            if row[index] == "NA":
                row[index] = col_avg


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        header = ["attribute", "min", "max", "mid", "avg", "median"]
        data = []
        for col in col_names:
            col_data = self.get_column(col, False)
            if col_data != []:
                curr = [col]
                curr.append(min(col_data))
                curr.append(max(col_data))
                curr.append((min(col_data)+max(col_data))/2)
                curr.append(sum(col_data)/len(col_data))
                col_data.sort()
                if len(col_data) % 2 == 0:
                    m1 = col_data[len(col_data)//2]
                    m2 = col_data[(len(col_data)//2) - 1]
                    curr.append((m1+m2)/2)
                else:
                    curr.append(col_data[len(col_data)//2])
                data.append(curr)
        return MyPyTable(header, data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        header = []
        data = []
        fullhead  = self.column_names + other_table.column_names
        key_pos_r = []
        for key in key_column_names:
            key_pos_r.append(other_table.column_names.index(key))
        for head in fullhead:
            if head in key_column_names and head in header:
                pass
            elif head in header:
                header.append(head + "(1)")
            else:
                header.append(head)
        l_keys = []
        r_keys = []
        key_indexes = []
        for row in self.data:
            keys = ""
            for key in key_column_names:
                keys += str(row[self.column_names.index(key)])
            l_keys.append(keys)
        for row in other_table.data:
            keys = ""
            for key in key_column_names:
                keys += str(row[other_table.column_names.index(key)])
            r_keys.append(keys)
        for i, key in enumerate(l_keys):
            indices = [i for i, x in enumerate(r_keys) if x == key]
            for j in indices:
                key_indexes.append((i,j))
        for left, right in key_indexes:
            new_row = []
            for col in self.data[left]:
                new_row.append(col)
            for col in other_table.data[right]:
                if other_table.data[right].index(col) in key_pos_r:
                    pass
                else:
                    new_row.append(col)
            data.append(new_row)
        return MyPyTable(header, data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        header = []
        data = []
        fullhead  = self.column_names + other_table.column_names
        key_pos_r = []
        for key in key_column_names:
            key_pos_r.append(other_table.column_names.index(key))
        for head in fullhead:
            if head in key_column_names and head in header:
                pass
            elif head in header:
                header.append(head + "(1)")
            else:
                header.append(head)
        l_keys = []
        r_keys = []
        key_indexes = []
        r_index = []
        l_index = []
        for row in self.data:
            keys = ""
            for key in key_column_names:
                keys += str(row[self.column_names.index(key)])
            l_keys.append(keys)
        for row in other_table.data:
            keys = ""
            for key in key_column_names:
                keys += str(row[other_table.column_names.index(key)])
            r_keys.append(keys)
        for i, key in enumerate(l_keys):
            if key in r_keys:
                indices = [i for i, x in enumerate(r_keys) if x == key]
                for j in indices:
                    key_indexes.append((i,j))
            else:
                l_index.append(i)
        for i, key in enumerate(r_keys):
            if key not in l_keys:
                r_index.append(i)
        for left, right in key_indexes:
            new_row = []
            for col in self.data[left]:
                new_row.append(col)
            for col in other_table.data[right]:
                if other_table.data[right].index(col) in key_pos_r:
                    pass
                else:
                    new_row.append(col)
            data.append(new_row)
        for i in l_index:
            new_row = []
            for col in self.data[i]:
                new_row.append(col)
            while len(new_row) != len(header):
                new_row.append("NA")
            data.append(new_row)
        for i in r_index:
            new_row = []
            for key in header:
                if key not in other_table.column_names:
                    new_row.append("NA")
                else:
                    new_row.append(other_table.data[i][other_table.column_names.index(key)])
            data.append(new_row)

        return MyPyTable(header, data)

