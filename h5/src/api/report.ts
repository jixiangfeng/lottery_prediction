import {
  normalizeReportData,
  normalizeWalkForwardStrategyReport,
  type ReportData,
  type WalkForwardStrategyReport,
} from '../domain/report'

export async function fetchLatestReport(issue?: string): Promise<ReportData> {
  const path = issue ? `/report-data/${issue}.json` : '/report-data/latest.json'
  const response = await fetch(path, { cache: 'no-store' })
  if (!response.ok) {
    throw new Error(`读取快乐8日报数据失败：${response.status}`)
  }
  return normalizeReportData(await response.json())
}

export async function fetchWalkForwardStrategyReport(): Promise<WalkForwardStrategyReport | null> {
  const response = await fetch('/report-data/walk_forward_kl8.json', { cache: 'no-store' })
  if (response.status === 404) {
    return null
  }
  if (!response.ok) {
    throw new Error(`读取快乐8前推回测数据失败：${response.status}`)
  }
  return normalizeWalkForwardStrategyReport(await response.json())
}
