"""Utilitários de formatação para números, percentuais e emojis de status."""

from __future__ import annotations


def formatar_numero(valor: float | int, casas: int = 4) -> str:
    """Formata número com separador de milhares e casas decimais.

    Ex.: ``formatar_numero(1234567.89, 2)`` → ``'1,234,567.89'``
    """
    try:
        if isinstance(valor, bool):
            return str(valor)
        if isinstance(valor, int):
            return f"{valor:,}"
        return f"{valor:,.{casas}f}"
    except Exception:
        return str(valor)


def formatar_percentual(valor: float, casas: int = 2) -> str:
    """Formata fração (0.1234) como percentual em texto (``'12.34%'``)."""
    return f"{valor * 100:.{casas}f}%"


def formatar_tempo(segundos: float) -> str:
    """Formata duração em segundos para texto legível (ms / s / min)."""
    if segundos >= 60.0:
        minutos = segundos / 60.0
        return f"{minutos:.1f} min"
    if segundos >= 1.0:
        return f"{segundos:.1f} s"
    return f"{segundos * 1000:.0f} ms"


def icone_status(status: str) -> str:
    """Retorna emoji conforme o status do item.

    status esperado: 'ok' | 'warn' | 'error' | 'none'.
    """
    if status == "ok":
        return "✅"
    if status == "warn":
        return "⚠️"
    if status == "error":
        return "❌"
    return "⭕"


def truncar(texto: str, limite: int = 120) -> str:
    """Corta o texto e adiciona reticências quando ultrapassar o limite."""
    if len(texto) <= limite:
        return texto
    return texto[: limite - 3] + "..."