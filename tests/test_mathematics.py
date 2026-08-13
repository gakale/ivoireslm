import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data.mathematics import clean_latex, rendered_math_html_to_text


def test_clean_latex_removes_displaystyle_wrapper_once():
    assert clean_latex(r"{\displaystyle x^2 + y^2}") == "$x^2 + y^2$"


def test_clean_latex_repairs_legacy_broken_math_boundaries():
    broken = r"a_n=o(u_n)/math>et<math>b_n=o(u_n)"
    assert clean_latex(broken) == "$a_n=o(u_n)$ et $b_n=o(u_n)$"


def test_rendered_math_html_keeps_text_and_single_latex_formula():
    html = r"""
    <h2>Définition</h2>
    <p>Une équation <math alttext="{\displaystyle x^2=4}"><mi>x</mi>
    <annotation encoding="application/x-tex">{\displaystyle x^2=4}</annotation></math>
    possède deux solutions.</p>
    <h2>Références</h2><p>Une adresse bibliographique à exclure.</p>
    """
    text = rendered_math_html_to_text(html)
    assert "Définition" in text
    assert "$x^2=4$" in text
    assert text.count("x^2=4") == 1
    assert "bibliographique" not in text
