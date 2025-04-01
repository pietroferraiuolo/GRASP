import os as _os
import sympy as _sp
from shutil import rmtree as _rm
from grasp._utility.base_formula import BaseFormula as _BaseFormula
from sympy.parsing import latex as _latex, sympy_parser as _symparser
from grasp.core.folder_paths import (
    SYS_DATA_FOLDER as _sdf,
    FORMULARY_BASE_FILE as _fbf,
)

_str2py = str.maketrans(
    {
        "-": "",
        " ": "_",
    }
)

_py2str = str.maketrans(
    {
        "_": " ",
    }
)

_tex2sym = {
    # Basic Operations
    "^": "**",
    "\\": "",  # Remove backslashes
}

_dummyF = ["f", "g", "h", "k"] * 50


class Formulary:
    """
    The formulary class, which is a collection of equations read from a file,
    or directly instanced in the python session.

    Parameters
    ----------
    name : str, optional
        The name of the formulary.
    formula_names : list, optional
        The names of the formulas passed to instance the formulary.
    formulas : list, optional
        The formulas passed to instance the formulary.

    """

    def __init__(self, name: str = "", formula_names: list = [], formulas: list = []):
        """The constructor"""
        self.name = name
        self.symbols = set({})
        self.functions = set({})
        self.formulas = dict(zip(formula_names, formulas))
        self._type = None
        self._file = None
        self._latexFormulas = {}
        self._add_count = 0
        super().__init__()

    def __len__(self):
        """Length"""
        return len(self.formulas)

    def __repr__(self):
        """Representation"""
        text = ""
        text += f"_{self.name}_ formulary"
        text += (
            f" from file '{self._file.split('/')[-1]}'\n"
            if self._file is not None
            else "\n"
        )
        text += f"Type: {self._type}\n"
        return text

    def __str__(self):
        """String representation"""
        text = ""
        text += f"Available formulas:\n"
        for i, name in enumerate(self.formulas.keys()):
            text += f"\n{i+1}. {name}"
        return text

    def __getitem__(self, key):
        """Item Getter"""
        return self.formulas[key]

    def __setitem__(self, key, value):
        """Item Setter"""
        self.formulas[key] = value

    def __getattr__(self, attr):
        """The attribute getter"""
        name = attr.translate(_py2str).title()
        if attr in self.formulas.keys() or name in self.formulas.keys():
            try:
                return self.formulas[name]
            except KeyError:
                return self.formulas[attr]
        else:
            raise AttributeError(f"No '{name}' formula found in the formulary.")

    def compute(
        self,
        name: str,
        data: dict,
        errors: dict = None,
        corrs: dict = None,
        asarray: bool = False,
    ):
        """
        Compute numerically a formula, given a set of data for each variable.

        Parameters
        ----------
        name : str
            The name of the formula to compute.
        data : dict
            The data to compute the formula. The format must be {'symbol': array}.
        errors : dict, optional
            The errors for each variable in the formula. The format must be
            {'symbol': array}.
        corrs : dict, optional
            The correlation between the variables. The format must be
            {'symbol1_symbol2': array}.

        Returns
        -------
        formula : _FormulaWrapper objcet
            An instance of the individual computed formula.

        """
        if (
            name or name.lower() or name.replace(" ", "_").lower()
        ) in self.formulas.keys():
            formula = (
                self.formulas[name].rhs
                if isinstance(self.formulas[name], _sp.Equality)
                else self.formulas[name]
            )
            variables = list(formula.free_symbols)
            if not all([v.name in data.keys() for v in variables]):
                raise ValueError("Missing data for some variables in the formula.")
            if errors is not None:
                if not all([v.name in errors.keys() for v in variables]):
                    raise ValueError(
                        "Missing errors for some variables in the formula."
                    )
                if corrs is not None:
                    if not all(
                        [
                            f"{v1.name}_{v2.name}" in corrs.keys()
                            for v1 in variables
                            for v2 in variables
                            if v1 != v2
                        ]
                    ):
                        raise ValueError(
                            "Missing correlations for some variables in the formula."
                        )
            formula = _FormulaWrapper(name, formula, variables)
            data_list = list(data.values())
            err_list = list(errors.values()) if errors is not None else None
            corr_list = list(corrs.values()) if corrs is not None else None
        else:
            raise ValueError(f"'{name}' not found in the formulary.")
        result = formula.compute(data_list, err_list, corr_list)
        if asarray:
            result = result.computed_values
        return result

    def substitute(self, name, values):
        """
        Substitute values to symbols in a formula.

        Parameters
        ----------
        name : str
            The name of the formula.
        values : dict
            The values to substitute in the formula. The format must be
            {'symbol': value}.

        """
        if (
            name or name.lower() or name.translate(_str2py).lower()
        ) in self.formulas.keys():
            formula = self.formulas[name]
            for v in values.values():
                if not isinstance(v, (int, float)):
                    raise ValueError("Values must be numerical.")
            if isinstance(formula, _sp.Equality):
                formula = formula.rhs
                formula = formula.subs(values)
                new_formula = _sp.Eq(self.formulas[name].lhs, formula)
                self.formulas[name] = new_formula
            else:
                self.formulas[name] = formula.subs(values)
        else:
            raise ValueError(f"'{name}' not found in the formulary.")

    def add_formula(self, name, formula):
        """
        Add a formula to the formulary.

        Parameters
        ----------
        name : str
            The name of the formula.
        formula : str
            The formula to add, which can be writte in latex or sympy syntax.

        """
        if isinstance(formula, str):
            if "Eq(" in formula:
                formula = _symparser.parse_expr(formula)
            else:
                try:
                    self._latexFormulas[name] = formula
                    l_formula = _latex.parse_latex(formula)
                    s_formula = _symparser.parse_expr(formula)
                    assert (
                        l_formula == s_formula
                    ), f"(latex)'{l_formula}' != '{s_formula}'(sympy)"
                except Exception as e:
                    if isinstance(e, AssertionError):
                        print(e)  # DEBUG
                        text = "Ambiguity in the written formula...\n"
                        try:  # Try to "translate" the formula from latex to sympy
                            transl = formula.translate(_tex2sym)
                            l2_formula = _symparser.parse_expr(transl)
                            assert (
                                l2_formula == s_formula
                            ), f"Can't translate '{formula}'"
                            formula = s_formula
                        except AssertionError as f:
                            print(f)
                        if isinstance((l_formula, s_formula), _sp.Equality):
                            l_formula = l_formula.rhs
                            s_formula = s_formula.rhs
                        if len(s_formula.free_symbols) >= len(l_formula.free_symbols):
                            formula = s_formula
                            text += "Assuming sympy syntax"
                        else:
                            formula = l_formula
                            text += "Assuming latex syntax"
                        print(text)
                    elif isinstance(e, (SyntaxError, ValueError, TypeError)):
                        formula = l_formula
                    elif isinstance(e, _latex.errors.LaTeXParsingError):
                        try:
                            formula = _symparser.parse_expr(formula)
                        except Exception as f:
                            raise e
                    else:
                        raise e
        if not isinstance(formula, _sp.Equality):
            formula = _sp.Eq(
                _sp.Symbol(f"{_dummyF[self._add_count]}_{self._add_count//4}"), formula
            )
            self._add_count += 1
        self.__setitem__(name, formula)
        self.symbols.update(formula.free_symbols)
        self.functions.update(formula.atoms(_sp.Function))

    def display_all(self):
        """
        Display all formulas in the current formulary instance.
        """
        for name, formula in self.formulas.items():
            if isinstance(formula, _sp.Equality):
                lhs, rhs = formula.args
                print(f"\n{name}\n{lhs} = {rhs}")
            else:
                print(f"\n{name}\n{formula}")

    def show_formula_symbols(self, name: str):
        """
        Show the symbols in a formula.

        Parameters
        ----------
        name : str
            The name of the formula.

        """
        if name in self.formulas.keys():
            formula = self.formulas[name]
            if isinstance(formula, _sp.Equality):
                symbols = formula.rhs.free_symbols
            else:
                symbols = formula.free_symbols
            print(f"Symbols in '{name}': {symbols}")
        else:
            raise ValueError(f"'{name}' not found in the formulary.")

    def load_formulary(self, filename: str):
        """
        Load a formulary from a file.

        If no filename is provided, the base formulary `Base Formulary.frm`
        will be loaded.

        Parameters
        ----------
        filename : str, optional
            The name of the file to load the formulary from.

        """
        if not ".frm" in filename:
            filename += ".frm"
        if len(filename.split("/")) == 1:
            filename = _os.path.join(_sdf, filename)
        self._file = filename
        self.name = filename.split("/")[-1].split(".")[0]
        with open(filename, "r") as frm:
            content = frm.readlines()
        self._type = content[0].split(":")[1].strip()
        for i in range(1, len(content), 2):
            if content[i] in ["\n", ""] or "#" in content[i]:
                continue
            name = content[i].strip()
            formula = content[i + 1].strip()
            if self._type == "latex":
                self.formulas[name] = _latex.parse_latex(formula)
                self._latexFormulas[name] = formula
            elif self._type == "sympy":
                self.formulas[name] = _symparser.parse_expr(formula)
            else:
                raise ValueError("Invalid formulary type: '%s'" % self._type)
            self.symbols.update(self.formulas[name].free_symbols)
            self.functions.update(self.formulas[name].atoms(_sp.Function))

    def update_formulary(self):
        """
        Updates the current loaded formulary file with the new formulae defined
        in the current instance, if any.

        """
        if self._file is None:
            raise ValueError("No file to update the formulary from.")
        self.save_formulary(self._file, self._type)

    def save_formulary(self, filename: str, type: str = "sympy"):
        """
        Save the formulary to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the formulary to.

        """
        if not ".frm" in filename:
            filename += ".frm"
        if len(filename.split("/")) == 1:
            filename = _os.path.join(_sdf, filename)
        with open(filename, "w") as frm:
            frm.write(f"type: {type}\n")
            if type == "sympy":
                for name, formula in self.formulas.items():
                    frm.write(f"{name}\n{str(formula)}\n")
            elif type == "latex":
                if not self._latexFormulas is None:
                    import warnings

                    warnings.warn(
                        "\nOnly the provided LaTeX formulas will be saved, "
                        "as sympy expressions can't be converted to LaTeX ones.",
                        UserWarning,
                    )
                    for name, formula in self._latexFormulas.items():
                        frm.write(f"{name}\n{formula}\n")
                else:
                    raise _latex.errors.LaTeXParsingError(
                        "No LaTeX formulas found to save."
                    )


def load_base_formulary():
    """
    Load the base formulary file.

    Returns
    -------
    formulary : Formulary object
        The formulary object containing the base formulary.

    """
    formulary = Formulary()
    formulary.load_formulary(_fbf)
    return formulary


class _FormulaWrapper(_BaseFormula):

    def __init__(self, name, formula, variables):
        """The constructor"""
        super().__init__()
        self._name = name
        self._formula = formula
        self._variables = variables
        self._values = None
        self._errors = None
        self._propagate_error()

    def _propagate_error(self):
        """
        Computes the analytical error for the quantity, through the
        error propagation formula.
        """
        from grasp.analyzers.calculus import error_propagation

        correlation = True if len(self._variables) > 1 else False
        propagation = error_propagation(self._formula, self._variables, correlation)
        self._errFormula = propagation["error_formula"]
        self._errVariables = propagation["error_variables"]["errors"]
        self._correlations = (
            propagation["error_variables"]["corrs"] if correlation else None
        )
