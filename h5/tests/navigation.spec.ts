import { describe, expect, it } from 'vitest'
import { navItems, routes } from '../src/router'
import { themeTokens } from '../src/theme'

describe('h5 navigation and theme', () => {
  it('defines four user-facing tabs', () => {
    expect(navItems.map((item) => item.label)).toEqual(['首页', '推荐', '回测', '策略'])
    expect(navItems.map((item) => item.to)).toEqual(['/', '/candidates', '/backtest', '/strategy'])
  })

  it('registers route components for all tabs', () => {
    expect(routes.map((route) => route.path)).toEqual(['/', '/report/:issue', '/candidates', '/backtest', '/strategy'])
  })

  it('includes a shareable report detail route', () => {
    expect(routes.find((route) => route.name === 'report-detail')?.path).toBe('/report/:issue')
  })

  it('uses blue-white visual tokens', () => {
    expect(themeTokens.primary).toMatch(/^#(1d4ed8|2563eb|0ea5e9)$/i)
    expect(themeTokens.background).toBe('#f4f8ff')
    expect(themeTokens.surface).toBe('#ffffff')
  })
})
