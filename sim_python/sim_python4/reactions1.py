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


@jit
def bound_inhibitor(fm_pre_con, fi_pre_con, im_pre_con, k_on, k_off, dt):
    """
    morphogen-inhibitor binding process
    Arguments:
        - fm_pre_con (float number): previous concentration of the free morphogen in a cell.
        - fi_pre_con (float number): previous concentration of the free inhibitor in a cell.
        - im_pre_con (float number): previous concentration of the inhibitor-morphogen in a cell.
        - k_on (float number): binding rate constant for the reaction m + i -> im.
        - k_off (float number): unbinding rate constant for the reaction m + i <- im.
        - dt (float number): time interval.
    Returns:
        - fm_con (float number): updated concentration of free morpgogen.
        - fi_con (float number): updated concentration of free inhibitor.
        - im_con (float number): updated concentration of inhibitor-morphogen.
    """
    fm_con = fm_pre_con + (k_off * im_pre_con * dt) - (k_on * fm_pre_con * fi_pre_con * dt)
    fi_con = fi_pre_con + (k_off * im_pre_con * dt) - (k_on * fm_pre_con * fi_pre_con * dt)
    im_con = im_pre_con + (k_on * fm_pre_con * fi_pre_con * dt) - (k_off * im_pre_con * dt)

    return fm_con, fi_con, im_con


@jit
def bound_anchor(fm_pre_con, am_pre_con, k_on, k_off, a_cells, dt):
    """
    anchor-morphogen binding process
    Arguments:
        - fm_pre_con (float number): previous concentration of the free morphogen in a cell.
        - am_pre_con (float number): previous concentration of the anchor-morphogen in a cell.
        - a_cells (float number): concentration of the anchor in a cell.
        - k_on (float number): binding rate constant for the reaction m + a -> am.
        - k_off (float number): unbinding rate constant for the reaction m + a <- am.
        - dt (float number): time interval.
    Returns:
        - fm_con (float number): updated concentration of free morpgogen.
        - fi_con (float number): updated concentration of free inhibitor.
        - im_con (float number): updated concentration of inhibitor-morphogen.
    """
    free_cells = np.maximum(0, a_cells - am_pre_con)
    am_con = am_pre_con + (k_on * fm_pre_con * free_cells * dt) - (k_off * am_pre_con * dt)
    fm_con = fm_pre_con - (k_on * fm_pre_con * free_cells * dt) + (k_off * am_pre_con * dt)

    return am_con, fm_con
