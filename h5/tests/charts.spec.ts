import { describe, expect, it } from 'vitest'
import { buildHitDistributionSeries, buildStrategyRoiSeries } from '../src/domain/charts'
import { normalizeReportData } from '../src/domain/report'

const report = normalizeReportData({
  schemaVersion: 1,
  lottery: 'kl8',
  play: 'select10',
  issue: '2026181',
  parameterName: 'omission_mix',
  latestNumbers: [],
  candidateGroups: [],
  backtest: {
    totalCost: 2000,
    totalPrize: 1313,
    roi: -0.3435,
    averageHit: 2.5,
    hitDistribution: { '0': 44, '1': 174, '5': 45, '10': 1 },
  },
  strategyComparison: {
    random: { strategy: 'random', summary: { totalCost: 2000, totalPrize: 600, roi: -0.7, averageHit: 2.1 } },
    omission_mix: { strategy: 'omission_mix', summary: { totalCost: 2000, totalPrize: 1313, roi: -0.3435, averageHit: 2.5 } },
  },
})

describe('chart domain', () => {
  it('builds hit distribution bar series from backtest summary', () => {
    const series = buildHitDistributionSeries(report)

    expect(series.labels).toEqual(['中0', '中1', '中5', '中10'])
    expect(series.values).toEqual([44, 174, 45, 1])
  })

  it('builds strategy roi series sorted by roi desc', () => {
    const series = buildStrategyRoiSeries(report)

    expect(series.labels).toEqual(['omission_mix', 'random'])
    expect(series.values).toEqual([-34.35, -70])
  })
})
