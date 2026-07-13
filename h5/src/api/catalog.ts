import { buildReportCatalog, type ReportCatalog } from '../domain/catalog'

export async function fetchReportCatalog(): Promise<ReportCatalog> {
  const response = await fetch('/report-data/index.json', { cache: 'no-store' })
  if (!response.ok) {
    throw new Error(`读取快乐8历史目录失败：${response.status}`)
  }
  return buildReportCatalog(await response.json())
}
