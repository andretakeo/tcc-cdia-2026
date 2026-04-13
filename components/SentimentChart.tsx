"use client";

import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  Line,
  ComposedChart,
} from "recharts";

interface ChartData {
  date: string;
  price: number;
  sentiment: number;
}

export function SentimentChart({ data }: { data: ChartData[] }) {
  return (
    <div className="h-[400px] w-full bg-white p-6 rounded-2xl border shadow-sm relative group overflow-hidden">
      <div className="absolute top-0 left-0 w-1 h-full bg-blue-600 opacity-0 group-hover:opacity-100 transition-opacity" />
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.1}/>
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />  
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11, fill: '#94a3b8' }}
            axisLine={false}
            tickLine={false}
            minTickGap={40}
          />
          <YAxis
            yAxisId="left"
            tick={{ fontSize: 11, fill: '#94a3b8' }}
            axisLine={false}
            tickLine={false}
            domain={['auto', 'auto']}
            name="PreÃ§o"
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fontSize: 11, fill: '#10b981' }}
            axisLine={false}
            tickLine={false}
            domain={[-1, 1]}
            name="Sentimento"
          />
          <Tooltip
            contentStyle={{
              borderRadius: '12px',
              border: 'none',
              boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
              padding: '12px'
            }}
          />
          <Area
            yAxisId="left"
            type="monotone"
            dataKey="price"
            stroke="#3b82f6"
            strokeWidth={3}
            fillOpacity={1}
            fill="url(#colorPrice)"
            name="PreÃ§o"
            animationDuration={1500}
          />
          <Line
            yAxisId="right"
            type="stepAfter"
            dataKey="sentiment"
            stroke="#10b981"
            strokeWidth={2}
            dot={{ r: 4, fill: '#10b981', strokeWidth: 2, stroke: '#fff' }}
            name="Sentimento AI"
            animationDuration={2000}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

