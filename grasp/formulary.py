"""
Author(s) 
---------
- Pietro Ferraiuolo : written in 2024

Description
-----------
"""
import os as _os
import sympy as _sp
from sympy.parsing import latex as _latex, sympy_parser as _symparser
from grasp._utility.base_formula import BaseFormula as _BaseFormula
from grasp.core.folder_paths import (
    SYS_DATA_FOLDER as _sdf,
    FORMULARY_BASE_FILE as _fbf,
)
from grasp import types as _gt

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
        text += "Available formulas:\n"
        for i, name in enumerate(self.formulas.keys()):
            text += f"\n{i+1}. {name}"
        return text

    def __getitem__(self, key):
        """Item Getter"""
        key = self._check_keys(key)
        return self.formulas[key]

    def __setitem__(self, key, value):
        """Item Setter"""
        self.formulas[key] = value

    def __getattr__(self, attr):
        """The attribute getter"""
        name = self._check_keys(attr)
        return self.formulas[name]
    
    @property
    def formula_names(self):
        """The formula names"""
        return list(self.formulas.keys())

    def compute(
        self,
        name: str,
        data: dict[str,_gt.ArrayLike],
        errors: dict[str,_gt.ArrayLike] = None,
        corrs: dict[str,_gt.ArrayLike] = None,
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
        name = self._check_keys(name)
        formula = (
            self.formulas[name].rhs
            if isinstance(self.formulas[name], _sp.Equality)
            else self.formulas[name]
        )
        variables = self._get_ordered_variables(formula)
        if not all([v.name in data.keys() for v in variables]):
            raise ValueError("Missing data for some variables in the formula.")
        if errors is not None:
            if not all([('epsilon_'+v.name) in errors.keys() for v in variables]):
                raise ValueError(
                    "Missing errors for some variables in the formula."
                )
            if corrs is not None:
                var_names = [v.name for v in variables]
                checked_pairs = set()
                for i, v1 in enumerate(var_names):
                    for v2 in var_names[i+1:]:
                        key1 = f"rho_{v1}_{v2}"
                        key2 = f"rho_{v2}_{v1}"
                        if key1 not in corrs and key2 not in corrs:
                            raise ValueError(
                                f"Missing correlation for variables: {v1}, {v2} in the formula."
                            )
                # if not all([
                #         f"rho_{v1.name}_{v2.name}" in corrs.keys()
                #         for v1 in variables
                #         for v2 in variables
                #         if v1 != v2
                # ]):
                #     raise ValueError(
                #         "Missing correlations for some variables in the formula."
                #     )
        formula = _FormulaWrapper(name.translate(_py2str).capitalize(), formula, variables)
        data_list = list(data.values())
        err_list = list(errors.values()) if errors is not None else None
        corr_list = list(corrs.values()) if corrs is not None else None
        result = formula.compute(data_list, err_list, corr_list)
        if asarray:
            if errors is not None:
                import numpy
                result = numpy.array([result.values.tolist(), result.errors.tolist()])
            else:
                result = result.computed_values
        return result

    def var_order(self, name: str):
        """
        Show the order the variables of a certain formula must have
        for the numerical computation.

        Parameters
        ----------
        name : str
            The name of the formula.
        """
        formula = self._get_formula(name)
        variables = self._get_ordered_variables(formula)
        formula = _FormulaWrapper(self._check_keys(name), formula, variables)
        formula.var_order()


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
        name = self._check_keys(name)
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
                            raise e from f
                    else:
                        raise e
        if not isinstance(formula, _sp.Equality):
            formula = _sp.Eq(
                _sp.Symbol(f"{_dummyF[self._add_count]}_{self._add_count//4}"), formula
            )
            self._add_count += 1
        self[name] = formula
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
        with open(filename, "r", encoding='utf-8') as frm:
            content = frm.readlines()
        self._type = content[0].split(":")[1].strip()
        for i in range(1, len(content), 2):
            if content[i] in ["\n", ""] or "#" in content[i]:
                continue
            name = (content[i].strip()).lower().capitalize()
            formula = content[i + 1].strip()
            if self._type == "latex":
                self.formulas[name] = _latex.parse_latex(formula)
                self._latexFormulas[name] = formula
            elif self._type == "sympy":
                self.formulas[name] = _symparser.parse_expr(formula)
            else:
                raise ValueError(f"Invalid formulary type: `{self._type}`")
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


    def save_formulary(self, filename: str, ftype: str = "sympy"):
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
        with open(filename, "w", encoding='utf-8') as frm:
            frm.write(f"type: {ftype}\n")
            if ftype == "sympy":
                for name, formula in self.formulas.items():
                    frm.write(f"{name}\n{str(formula)}\n")
            elif ftype == "latex":
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


    def _check_keys(self, name):
        """Check if the keys are valid"""
        fk = list(self.formulas.keys())
        fk_lower = [k.lower() for k in fk]
        fk_l_us = [k.translate(_str2py) for k in fk_lower]
        fk_l_ns = [k.translate(_py2str) for k in fk_lower]
        fk_cap = [k.capitalize() for k in fk_lower]
        fk_tit = [k.title() for k in fk_lower]
        if name in fk:
            pass
        elif name in fk_lower:
            name = name.capitalize()
        elif name in fk_cap:
            name = name.lower().capitalize()
        elif name in fk_tit:
            name = name.lower().capitalize()
        elif name in fk_l_us:
            name = name.translate(_py2str).capitalize()
        elif name in fk_l_ns:
            name = name.translate(_py2str).lower().capitalize()
        else:
            raise ValueError(f"'{name}' not found in the formulary.")
        return name


    def _get_ordered_variables(self, formula):
        """
        Get the ordered variables in a formula.

        Parameters
        ----------
        formula : sympy expression
            The formula to get the variables from.

        Returns
        -------
        variables : list
            The ordered variables in the formula.
        """
        variables = list(formula.free_symbols)
        variables = sorted(variables, key=lambda x: str(x))
        return variables


    def _get_formula(self, name):
        """
        Get a formula from the formulary.

        Parameters
        ----------
        name : str
            The name of the formula.

        Returns
        -------
        formula : sympy expression
            The formula.

        """
        name = self._check_keys(name)
        formula = (
            self.formulas[name].rhs
            if isinstance(self.formulas[name], _sp.Equality)
            else self.formulas[name]
        )
        return formula


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

        correlation = len(self._variables) > 1
        propagation = error_propagation(self._formula, self._variables, correlation)
        self._errFormula = propagation["error_formula"]
        self._errVariables = propagation["error_variables"]["errors"]
        self._correlations = (
            propagation["error_variables"]["corrs"] if correlation else None
        )

    def var_order(self):
        """
        Returns the order of the variables in the formula.
        """
        print(f"""
`{self._name}' variables must be passed in the following order:
Data         : {self.variables}
Errors       : {self.error_variables}
Correlations : {self.correlations}""")
