import { topStrategies, type ReportData } from './report'

export interface ChartSeries {
  labels: string[]
  values: number[]
}

export function buildHitDistributionSeries(report: ReportData): ChartSeries {
  const distribution = report.backtest?.hitDistribution ?? {}
  const entries = Object.entries(distribution).sort(([left], [right]) => Number(left) - Number(right))
  return {
    labels: entries.map(([hit]) => `中${hit}`),
    values: entries.map(([, count]) => Number(count)),
  }
}

export function buildStrategyRoiSeries(report: ReportData): ChartSeries {
  const strategies = topStrategies(report)
  return {
    labels: strategies.map((item) => item.strategy),
    values: strategies.map((item) => Number(((item.summary?.roi ?? 0) * 100).toFixed(2))),
  }
}
