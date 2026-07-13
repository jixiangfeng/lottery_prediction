import { describe, expect, it } from 'vitest'
import { buildReportCatalog, selectReportIssue } from '../src/domain/catalog'

const index = {
  latestIssue: '2026181',
  reports: [
    { issue: '2026181', label: '2026-181', path: '/report-data/2026181.json' },
    { issue: '2026180', label: '2026-180', path: '/report-data/2026180.json' },
  ],
}

describe('report catalog', () => {
  it('builds stable catalog and defaults to latest issue', () => {
    const catalog = buildReportCatalog(index)
    expect(catalog.latestIssue).toBe('2026181')
    expect(catalog.reports).toHaveLength(2)
    expect(catalog.reports[0].label).toBe('2026-181')
  })

  it('selects the requested issue when available', () => {
    const catalog = buildReportCatalog(index)
    expect(selectReportIssue(catalog, '2026180')).toBe('2026180')
    expect(selectReportIssue(catalog, 'missing')).toBe('2026181')
  })
})
