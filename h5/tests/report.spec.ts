import { describe, expect, it } from 'vitest'
import { topStrategies, normalizeReportData, normalizeWalkForwardStrategyReport, topWalkForwardSummaries } from '../src/domain/report'

const baseReport = {
  schemaVersion: 1,
  lottery: 'kl8',
  play: 'select10',
  issue: '2026181',
  parameterName: 'omission_mix',
  latestNumbers: [1, 2, 3],
  candidateGroups: [
    { rank: 1, numbers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], score: 88, oddCount: 5, evenCount: 5, bigCount: 0, smallCount: 10, repeatLastCount: 2, zoneDistribution: [10, 0, 0, 0] },
  ],
  backtest: { totalCost: 2000, totalPrize: 1313, roi: -0.3435, averageHit: 2.5 },
  strategyComparison: {
    random: { strategy: 'random', summary: { roi: -0.7, totalPrize: 600 } },
    omission_mix: { strategy: 'omission_mix', summary: { roi: -0.34, totalPrize: 1313 } },
  },
  slidingWindow: {},
  parameterSearch: [{ rank: 1, name: 'omission_mix', score: -0.2 }],
  walkForwardParameterWeights: { enabled: true, alpha: 0.08, matchedCount: 8, bestStrategy: 'omission_mix' },
  bettingPlan: {
    play: 'kl8_select10',
    displayName: '快乐8选十',
    coreNumbers: [1, 2, 3, 4],
    assistNumbers: [5, 6],
    defensiveNumbers: [],
    budgets: [20, 150],
    plans: [{ kind: '复式', title: '选十12码复式', numbers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], betCount: 66, cost: 132, riskLevel: '中', note: '预算内' }],
  },
  artifacts: { html: 'reports/html/kl8_daily_2026181.html', pickSnapshot: 'reports/picks/kl8_2026182.json' },
  disclaimer: '仅供娱乐',
}

describe('report domain', () => {
  it('normalizes raw report json for H5 rendering', () => {
    const report = normalizeReportData(baseReport)

    expect(report.issue).toBe('2026181')
    expect(report.latestNumbers).toEqual([1, 2, 3])
    expect(report.candidateGroups[0].numbers).toHaveLength(10)
    expect(report.backtest?.roiPercent).toBe('-34.35%')
    expect(report.walkForwardParameterWeights?.bestStrategy).toBe('omission_mix')
    expect(report.bettingPlan?.plans[0].cost).toBe(132)
  })

  it('sorts strategies by roi desc', () => {
    const report = normalizeReportData(baseReport)

    expect(topStrategies(report).map((item) => item.strategy)).toEqual(['omission_mix', 'random'])
  })

  it('normalizes and sorts walk-forward strategy summaries by score', () => {
    const walkForward = normalizeWalkForwardStrategyReport({
      schemaVersion: 1,
      lottery: 'kl8',
      play: 'select10',
      periodCount: 300,
      minTrainSize: 200,
      groupCount: 10,
      ticketPrice: 2,
      bestStrategy: 'omission_mix',
      summaries: [
        { strategy: 'hot', score: -0.9, roi: -0.8, averageHit: 2.4, hit5PlusCount: 10, hit6PlusCount: 1, issueHit5PlusCount: 8, maxLosingStreak: 5, bestIssue: '1', bestIssueMaxHit: 7, recentRoi: -0.7, periodCount: 300, groupCount: 10, totalCost: 6000, totalPrize: 1000 },
        { strategy: 'omission_mix', score: -0.6, roi: -0.5, averageHit: 2.6, hit5PlusCount: 12, hit6PlusCount: 2, issueHit5PlusCount: 9, maxLosingStreak: 4, bestIssue: '2', bestIssueMaxHit: 8, recentRoi: -0.6, periodCount: 300, groupCount: 10, totalCost: 6000, totalPrize: 2000 },
      ],
    })

    expect(topWalkForwardSummaries(walkForward).map((item) => item.strategy)).toEqual(['omission_mix', 'hot'])
  })
})
