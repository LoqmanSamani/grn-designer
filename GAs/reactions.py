import numpy as np
from numba import jit


@jit
def production(pre_con, num_cell, pk, dt):
    """
    morphogen/ inhibitor production process
    Updates the concentration of the morphogen/inhibitor from pre_con (previous concentration) to con (concentration).
    Arguments:
        - pre_con (float number): previous concentration of the morphogen/inhibitor in a cell.
        - num_cell (int or float): cell number (I think it is the initial concentration of product in corresponding cells !!!).
        - pk (float number): production rate constant of the morphogen/inhibitor.
        - dt (float number): time interval.
    Returns:
        - con (float number): updated concentration of free morpgogen/inhibitor.
    """
    con = pre_con + (pk * num_cell * dt)

    return con

@jit
def production1(pre_con, num_cell, fm, pk, dt):
    """
    morphogen/ inhibitor production process
    Updates the concentration of the morphogen/inhibitor from pre_con (previous concentration) to con (concentration).
    Arguments:
        - pre_con (float number): previous concentration of the morphogen/inhibitor in a cell.
        - num_cell (int or float): cell number (I think it is the initial concentration of product in corresponding cells !!!).
        - pk (float number): production rate constant of the morphogen/inhibitor.
        - dt (float number): time interval.
    Returns:
        - con (float number): updated concentration of free morpgogen/inhibitor.
    """
    con = pre_con + (pk * fm * num_cell * dt)

    return con


@jit
def degradation(pre_con, dk, dt):
    """
    morphogen/inhibitor degradation process
    Updates the concentration of the morphogen/inhibitor from pre_con (previous concentration) to con (concentration).
    Arguments:
        - pre_con (float number): previous concentration of the morphogen/inhibitor in a cell.
        - dk (float number): degradation rate constant of the Morphogen/inhibitor.
        - dt (float number): time interval.
    Returns:
        - con (float number): updated concentration of free morpgogen/inhibitor.
    """

    con = pre_con - (dk * pre_con * dt)

    return con


