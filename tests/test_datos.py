import numpy as np
import pandas as pd

from ocasion.datos import a_entero, limpiar


def test_a_entero_formatos_espanoles():
    assert a_entero("12.500 €") == 12500
    assert a_entero("85.000 km") == 85000
    assert a_entero("110 CV") == 110
    assert a_entero(2016) == 2016
    assert np.isnan(a_entero(None))


def test_limpiar_normaliza_y_filtra():
    df = pd.DataFrame(
        {
            "marca": ["SEAT", "Seat", "Kia"],
            "modelo": ["León", "León", "Ceed"],
            "anio": [2016, 2016, 2035],
            "km": ["85.000 km", "85.000 km", "10"],
            "combustible": ["Diésel", "Diésel", "Gasolina"],
            "cambio": ["Manual", "Manual", "Manual"],
            "potencia_cv": [np.nan] * 3,
            "precio": ["11.900 €", "11.900 €", "9.000 €"],
        }
    )
    limpio = limpiar(df, anio_actual=2026)
    assert len(limpio) == 1  # duplicado eliminado y año imposible fuera
    fila = limpio.iloc[0]
    assert (fila["marca"], fila["modelo"], fila["combustible"], fila["precio"]) == (
        "seat",
        "leon",
        "diesel",
        11900,
    )
