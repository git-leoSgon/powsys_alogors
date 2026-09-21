import math
import time

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition

def resolver_fpo(
    DBAR, DLIN, DGER, DDEF, cenario, pmax_mmgd,
    capacidade_bess, perdas, config,
):
    NB = len(DBAR)
    NL = len(DLIN)
    NG = len(DGER)
    ND = len(DDEF)
    NGD = len(pmax_mmgd)
    NBAT = len(capacidade_bess)

    tap = DLIN["Tap"].to_numpy()
    bs = 1.0 / (DLIN["X"].to_numpy() * tap)
    gs = DLIN["R"].to_numpy() / (DLIN["R"].to_numpy() ** 2 + DLIN["X"].to_numpy() ** 2) / tap**2

    model = pyo.ConcreteModel("FPO-DC multiperiodo com BESS")
    model.B = pyo.Set(initialize=range(NB))
    model.L = pyo.Set(initialize=range(NL))
    model.G = pyo.Set(initialize=range(NG))
    model.D = pyo.Set(initialize=range(ND))
    model.GD = pyo.Set(initialize=range(NGD))
    model.BA = pyo.Set(initialize=range(NBAT))
    model.T = pyo.Set(initialize=range(config.DT))
    model.TSOC = pyo.Set(initialize=range(config.DT + 1))

    model.Th = pyo.Var(model.B, model.T, bounds=(-math.pi, math.pi))
    model.Pg = pyo.Var(
        model.G, model.T,
        bounds=lambda m, g, h: (DGER.loc[g, "Pmin"], DGER.loc[g, "Pmax"]),
    )
    model.Fd = pyo.Var(
        model.L, model.T,
        bounds=lambda m, l, h: (-DLIN.loc[l, "Flim"], DLIN.loc[l, "Flim"]),
    )
    model.Fc = pyo.Var(
        model.L, model.T,
        bounds=lambda m, l, h: (-DLIN.loc[l, "Flim"], DLIN.loc[l, "Flim"]),
    )
    model.Pcut = pyo.Var(
        model.GD, model.T,
        bounds=lambda m, gd, h: (0.0, pmax_mmgd[gd] * cenario["Fcap"][h, gd]),
    )
    model.Pdef = pyo.Var(model.D, model.T, within=pyo.NonNegativeReals)
    model.Pbin = pyo.Var(
        model.BA, model.T,
        bounds=lambda m, b, h: (0.0, capacidade_bess[b]),
    )
    model.Pbout = pyo.Var(
        model.BA, model.T,
        bounds=lambda m, b, h: (0.0, capacidade_bess[b]),
    )
    model.Soc = pyo.Var(
        model.BA, model.TSOC, bounds=(config.SOC_MIN, config.SOC_MAX)
    )
    model.DeltaPg = pyo.Var(model.G, model.T, within=pyo.NonNegativeReals)
    model.DeltaPbat = pyo.Var(model.BA, model.T, within=pyo.NonNegativeReals)
    model.PicoConv = pyo.Var(within=pyo.NonNegativeReals)
    peso_participacao = getattr(config, "PESO_PARTICIPACAO", 0.0)
    if peso_participacao > 0:
        model.DesvioParticipacao = pyo.Var(
            model.G, model.T, within=pyo.NonNegativeReals
        )

    custo_geracao = pyo.quicksum(
        DGER.loc[g, "Custo"] * model.Pg[g, h] for g in model.G for h in model.T
    )
    custo_corte = pyo.quicksum(
        400.0 * model.Pcut[gd, h] for gd in model.GD for h in model.T
    )
    custo_deficit = pyo.quicksum(
        DDEF.loc[d, "Custo"] * model.Pdef[d, h] for d in model.D for h in model.T
    )
    custo_bess = config.CUSTO_BESS * pyo.quicksum(
        model.Pbin[b, h] + model.Pbout[b, h] for b in model.BA for h in model.T
    )
    custo_variacao_geradores = config.PESO_VARIACAO_COM_BESS * pyo.quicksum(
        model.DeltaPg[g, h] for g in model.G for h in model.T
    )
    custo_pico = config.PESO_PICO * model.PicoConv
    custo_variacao_bess = config.PESO_VARIACAO_BESS * pyo.quicksum(
        model.DeltaPbat[b, h] for b in model.BA for h in model.T
    )
    custo_participacao = 0.0
    if peso_participacao > 0:
        custo_participacao = peso_participacao * pyo.quicksum(
            model.DesvioParticipacao[g, h] for g in model.G for h in model.T
        )
    desempate = 1e-4 * pyo.quicksum(
        g * model.Pg[g, h] for g in model.G for h in model.T
    )
    model.obj = pyo.Objective(
        expr=(
            custo_geracao + custo_corte + custo_deficit + custo_bess
            + custo_variacao_geradores + custo_pico + custo_variacao_bess
            + custo_participacao + desempate
        ),
        sense=pyo.minimize,
    )

    model.balanco = pyo.ConstraintList()
    for h in model.T:
        for i in model.B:
            pg = pyo.quicksum(model.Pg[g, h] for g in model.G if DGER.loc[g, "Barra"] == i)
            pdef = pyo.quicksum(model.Pdef[d, h] for d in model.D if DDEF.loc[d, "Barra"] == i)
            pgd = pyo.quicksum(
                pmax_mmgd[gd] * cenario["Fcap"][h, gd] - model.Pcut[gd, h]
                for gd in model.GD if cenario["barras_mmgd"][gd] == i
            )
            pbat = pyo.quicksum(
                model.Pbout[b, h] - model.Pbin[b, h]
                for b in model.BA if cenario["barras_mmgd"][b] == i
            )
            fluxo = pyo.quicksum(model.Fd[l, h] for l in model.L if DLIN.loc[l, "De"] == i)
            fluxo += pyo.quicksum(model.Fc[l, h] for l in model.L if DLIN.loc[l, "Para"] == i)
            pd = DBAR.loc[i, "Pd"] * cenario["Fcar"][h, i]
            model.balanco.add(pg + pdef + pgd + pbat - pd == fluxo)

    model.limite_deficit = pyo.ConstraintList()
    for d in model.D:
        barra = int(DDEF.loc[d, "Barra"])
        for h in model.T:
            model.limite_deficit.add(
                model.Pdef[d, h] <= DBAR.loc[barra, "Pd"] * cenario["Fcar"][h, barra]
            )

    if peso_participacao > 0:
        pmax_total = DGER["Pmax"].sum()
        model.participacao = pyo.ConstraintList()
        for g in model.G:
            fator = DGER.loc[g, "Pmax"] / pmax_total
            for h in model.T:
                referencia = fator * pyo.quicksum(model.Pg[j, h] for j in model.G)
                model.participacao.add(
                    model.DesvioParticipacao[g, h] >= model.Pg[g, h] - referencia
                )
                model.participacao.add(
                    model.DesvioParticipacao[g, h] >= referencia - model.Pg[g, h]
                )

    model.soc_inicial = pyo.Constraint(
        model.BA, rule=lambda m, b: m.Soc[b, 0] == config.SOC_INICIAL
    )

    def estado_bateria(m, b, h):
        if capacidade_bess[b] <= 0:
            return m.Soc[b, h + 1] == m.Soc[b, h]
        return m.Soc[b, h + 1] == m.Soc[b, h] + (
            config.ETA_CARGA * m.Pbin[b, h]
            - m.Pbout[b, h] / config.ETA_DESCARGA
        ) / capacidade_bess[b]

    model.estado_bateria = pyo.Constraint(model.BA, model.T, rule=estado_bateria)
    model.soc_final = pyo.Constraint(
        model.BA,
        rule=lambda m, b: m.Soc[b, config.DT] == config.SOC_INICIAL,
    )

    model.carga_solar = pyo.Constraint(
        model.BA, model.T,
        rule=lambda m, b, h: m.Pbin[b, h]
        <= pmax_mmgd[b] * cenario["Fcap"][h, b] - m.Pcut[b, h],
    )

    model.limite_pico = pyo.Constraint(
        model.T,
        rule=lambda m, h: pyo.quicksum(m.Pg[g, h] for g in m.G) <= m.PicoConv,
    )

    model.variacao_bateria = pyo.ConstraintList()
    for b in model.BA:
        for h in model.T:
            anterior = (h - 1) % config.DT
            potencia = model.Pbout[b, h] - model.Pbin[b, h]
            potencia_anterior = model.Pbout[b, anterior] - model.Pbin[b, anterior]
            model.variacao_bateria.add(model.DeltaPbat[b, h] >= potencia - potencia_anterior)
            model.variacao_bateria.add(model.DeltaPbat[b, h] >= potencia_anterior - potencia)
            model.variacao_bateria.add(
                model.DeltaPbat[b, h] <= config.RAMPA_BESS * capacidade_bess[b]
            )

    model.rampa_geradores = pyo.ConstraintList()
    for g in model.G:
        limite = config.RAMPA_GERADORES * DGER.loc[g, "Pmax"]
        for h in model.T:
            anterior = (h - 1) % config.DT
            model.rampa_geradores.add(model.Pg[g, h] - model.Pg[g, anterior] <= limite)
            model.rampa_geradores.add(model.Pg[g, anterior] - model.Pg[g, h] <= limite)
            model.rampa_geradores.add(model.DeltaPg[g, h] >= model.Pg[g, h] - model.Pg[g, anterior])
            model.rampa_geradores.add(model.DeltaPg[g, h] >= model.Pg[g, anterior] - model.Pg[g, h])

    def fluxo_direto(m, l, h):
        de = int(DLIN.loc[l, "De"])
        para = int(DLIN.loc[l, "Para"])
        fase = math.radians(DLIN.loc[l, "Phs"])
        return m.Fd[l, h] == bs[l] * (m.Th[de, h] - m.Th[para, h] - fase) + 0.5 * perdas[l, h]

    def fluxo_reverso(m, l, h):
        de = int(DLIN.loc[l, "De"])
        para = int(DLIN.loc[l, "Para"])
        fase = math.radians(DLIN.loc[l, "Phs"])
        return m.Fc[l, h] == bs[l] * (m.Th[para, h] - m.Th[de, h] + fase) + 0.5 * perdas[l, h]

    model.fluxo_direto = pyo.Constraint(model.L, model.T, rule=fluxo_direto)
    model.fluxo_reverso = pyo.Constraint(model.L, model.T, rule=fluxo_reverso)

    referencia = int(DBAR.index[DBAR["Tipo"] == 2][0])
    model.referencia = pyo.Constraint(
        model.T, rule=lambda m, h: m.Th[referencia, h] == 0.0
    )

    resultado = SolverFactory("glpk", executable=config.SOLVER).solve(model)
    if resultado.solver.termination_condition != TerminationCondition.optimal:
        raise RuntimeError("Solucao nao encontrada.")

    Ang = np.array([[pyo.value(model.Th[i, h]) for h in model.T] for i in model.B])
    Pg = np.array([[pyo.value(model.Pg[g, h]) for h in model.T] for g in model.G])
    Fd = np.array([[pyo.value(model.Fd[l, h]) for h in model.T] for l in model.L])
    Fc = np.array([[pyo.value(model.Fc[l, h]) for h in model.T] for l in model.L])
    Pcut = np.array([[pyo.value(model.Pcut[gd, h]) for h in model.T] for gd in model.GD])
    Pdef = np.array([[pyo.value(model.Pdef[d, h]) for h in model.T] for d in model.D])
    Pbin = np.array([[pyo.value(model.Pbin[b, h]) for h in model.T] for b in model.BA])
    Pbout = np.array([[pyo.value(model.Pbout[b, h]) for h in model.T] for b in model.BA])
    Soc = np.array([[pyo.value(model.Soc[b, h]) for h in model.TSOC] for b in model.BA])

    perdas_novas = np.zeros((NL, config.DT))
    for l in range(NL):
        de = int(DLIN.loc[l, "De"])
        para = int(DLIN.loc[l, "Para"])
        perdas_novas[l] = gs[l] * (Ang[de] - Ang[para]) ** 2

    return {
        "Fob": pyo.value(model.obj), "Ang": Ang, "Pg": Pg,
        "Fd": Fd, "Fc": Fc, "Pcut": Pcut, "Pdef": Pdef,
        "Pbin": Pbin, "Pbout": Pbout, "Soc": Soc,
        "Ploss": perdas_novas,
    }


def resolver_com_perdas(
    DBAR, DLIN, DGER, DDEF, cenario, pmax_mmgd, capacidade_bess, config,
):
    perdas = np.zeros((len(DLIN), config.DT))
    inicio = time.time()

    for iteracao in range(1, 101):
        solucao = resolver_fpo(
            DBAR, DLIN, DGER, DDEF, cenario, pmax_mmgd,
            capacidade_bess, perdas, config,
        )
        erro = np.max(np.abs(solucao["Ploss"] - perdas)) * config.PB
        if getattr(config, "MOSTRAR_ITERACOES", False) and (
            iteracao == 1 or iteracao % 10 == 0 or erro <= config.TOLERANCIA_PERDAS
        ):
            print(f"  perdas: iteracao {iteracao:3d} | erro {erro:9.4f} MW")

        if erro <= config.TOLERANCIA_PERDAS:
            solucao = resolver_fpo(
                DBAR, DLIN, DGER, DDEF, cenario, pmax_mmgd,
                capacidade_bess, solucao["Ploss"], config,
            )
            solucao["iteracoes"] = iteracao
            solucao["erro_perdas_MW"] = erro
            solucao["tempo"] = time.time() - inicio
            return solucao

        perdas = (
            (1.0 - config.RELAXACAO_PERDAS) * perdas
            + config.RELAXACAO_PERDAS * solucao["Ploss"]
        )

    raise RuntimeError("Limite de iteracoes atingido.")
