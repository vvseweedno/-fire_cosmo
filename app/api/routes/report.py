# Report API routes
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse


router = APIRouter(prefix="/report", tags=["Reports"])


@router.get("/{event_id}", response_class=HTMLResponse)
async def get_report_html(event_id: str):
    """Получить HTML-отчет о событии"""
    
    html = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Отчет о пожаре {event_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #f44336; color: white; padding: 20px; border-radius: 8px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 4px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
            .metric-label {{ font-size: 12px; color: #666; }}
            .severity-high {{ color: #d32f2f; }}
            .severity-moderate {{ color: #f57c00; }}
            .severity-low {{ color: #fbc02d; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            .warning {{ background: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔥 Отчет о лесном пожаре</h1>
            <p>ID события: {event_id}</p>
        </div>
        
        <div class="section">
            <h2>📊 Основная информация</h2>
            <div class="metric">
                <div class="metric-value">125.5</div>
                <div class="metric-label">Площадь (га)</div>
            </div>
            <div class="metric">
                <div class="metric-value severity-moderate">Moderate</div>
                <div class="metric-label">Степень поражения</div>
            </div>
            <div class="metric">
                <div class="metric-value">High</div>
                <div class="metric-label">Уверенность</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🛰 Данные Sentinel-2</h2>
            <table>
                <tr>
                    <th>Снимок</th>
                    <th>ID сцены</th>
                    <th>Дата</th>
                    <th>Облачность</th>
                </tr>
                <tr>
                    <td>До пожара</td>
                    <td>S2A_MSIL2A_20240101...</td>
                    <td>2024-01-01</td>
                    <td>5.2%</td>
                </tr>
                <tr>
                    <td>После пожара</td>
                    <td>S2B_MSIL2A_20240115...</td>
                    <td>2024-01-15</td>
                    <td>8.7%</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>📈 Степень поражения по площади</h2>
            <table>
                <tr>
                    <th>Категория</th>
                    <th>Площадь (га)</th>
                    <th>% от общей</th>
                </tr>
                <tr>
                    <td>Низкая</td>
                    <td>25.0</td>
                    <td>19.9%</td>
                </tr>
                <tr>
                    <td>Средняя</td>
                    <td>75.5</td>
                    <td>60.2%</td>
                </tr>
                <tr>
                    <td>Высокая</td>
                    <td>25.0</td>
                    <td>19.9%</td>
                </tr>
            </table>
        </div>
        
        <div class="warning">
            <strong>⚠️ Ограничения:</strong>
            <ul>
                <li>Расчет выполнен на основе фикстурных данных</li>
                <li>Требуется верификация по наземным данным</li>
            </ul>
        </div>
        
        <div class="section">
            <p><small>Сгенерировано: {__import__('datetime').datetime.utcnow().isoformat()} UTC</small></p>
        </div>
    </body>
    </html>
    """
    
    return html
