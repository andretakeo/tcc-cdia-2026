import { test, expect } from '@playwright/test';

test.describe('Quantitative Pipeline E2E', () => {
  
  test('should load dashboard and display market data', async ({ page }) => {
    await page.goto('/');
    
    // Check main headers
    await expect(page.locator('h1')).toContainText('ITUB4');
    await expect(page.getByText('Market Sentiment Correlation')).toBeVisible();
    
    // Check if chart container exists
    const chart = page.locator('.recharts-responsive-container');
    await expect(chart).toBeVisible();
  });

  test('should trigger news refresh and update feed', async ({ page }) => {
    await page.goto('/');
    
    const refreshButton = page.getByRole('button', { name: /Atualizar Sinal/i });
    await expect(refreshButton).toBeEnabled();
    
    // Trigger refresh
    await refreshButton.click();
    
    // Wait for "Triando Notícias" state or success reload
    await expect(page.getByText(/Triando Notícias/i)).toBeVisible();
    
    // After reload (triggered by button logic), check if articles appear
    await page.waitForLoadState('networkidle');
    const articles = page.locator('a[href^="/article/"]');
    // Ensure we have at least some articles (either from seed or refresh)
    await expect(articles.first()).toBeVisible();
  });

  test('should navigate to article details and audit AI rationale', async ({ page }) => {
    await page.goto('/');
    
    // Click first article
    const firstArticle = page.locator('a[href^="/article/"]').first();
    const title = await firstArticle.locator('h3').textContent();
    await firstArticle.click();
    
    // Verify detail page content
    await expect(page).toHaveURL(/\/article\//);
    await expect(page.locator('h1')).toContainText(title || '');
    await expect(page.getByText(/Auditoria de IA/i)).toBeVisible();
    
    // Check for sentiment badges
    const triageCard = page.getByText(/Triagem/i);
    await expect(triageCard).toBeVisible();
  });

  test('should display execution logs correctly', async ({ page }) => {
    await page.goto('/logs');
    
    await expect(page.locator('h1')).toContainText(/Histórico de Execuções/i);
    
    // Check table headers
    await expect(page.getByRole('columnheader', { name: /Data\/Hora/i })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: /Estado/i })).toBeVisible();
    
    // Ensure logs are present (since we seeded/refreshed)
    const logRows = page.locator('tbody tr');
    await expect(logRows.first()).toBeVisible();
  });

});
