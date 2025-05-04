import React, { useState, useEffect } from 'react';
import { Box, Card, CardContent, Typography, LinearProgress, Chip } from '@mui/material';
import { getProcessingStatus } from '../services/api';

const ProcessingStatus = () => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await getProcessingStatus();
        setStatus(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching processing status:', error);
        setLoading(false);
      }
    };

    fetchStatus();
    
    // Set up polling if processing is active
    if (status?.is_processing) {
      const interval = setInterval(fetchStatus, 2000);
      return () => clearInterval(interval);
    }
  }, [status?.is_processing]);

  if (loading) {
    return null;
  }

  if (!status || !status.is_processing) {
    return null;
  }

  const progress = status.total_tickets > 0 
    ? Math.round((status.processed_tickets / status.total_tickets) * 100)
    : 0;

  return (
    <Box sx={{ mb: 3 }}>
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Processing Status
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography variant="body2" sx={{ mr: 2 }}>
              {status.processed_tickets} of {status.total_tickets} tickets processed
            </Typography>
            <Chip
              label={status.is_processing ? 'In Progress' : 'Completed'}
              color={status.is_processing ? 'primary' : 'success'}
              size="small"
            />
          </Box>
          
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{ height: 8, borderRadius: 4 }}
          />
          
          {status.estimated_completion && (
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
              {status.estimated_completion}
            </Typography>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default ProcessingStatus;