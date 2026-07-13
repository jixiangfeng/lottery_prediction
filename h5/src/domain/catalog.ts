export interface ReportCatalogItem {
  issue: string
  label: string
  path: string
}

export interface ReportCatalog {
  latestIssue: string
  reports: ReportCatalogItem[]
}

export function buildReportCatalog(raw: unknown): ReportCatalog {
  const data = raw as Partial<ReportCatalog>
  const reports = (data.reports ?? [])
    .filter((item): item is ReportCatalogItem => Boolean(item?.issue && item?.path))
    .map((item) => ({
      issue: String(item.issue),
      label: String(item.label ?? item.issue),
      path: String(item.path),
    }))
    .sort((left, right) => Number(right.issue) - Number(left.issue))

  const latestIssue = String(data.latestIssue ?? reports[0]?.issue ?? '')
  return { latestIssue, reports }
}

export function selectReportIssue(catalog: ReportCatalog, issue?: string | null): string {
  if (!issue) return catalog.latestIssue
  return catalog.reports.some((item) => item.issue === issue) ? issue : catalog.latestIssue
}
