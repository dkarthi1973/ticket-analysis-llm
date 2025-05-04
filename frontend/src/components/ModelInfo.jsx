import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Divider,
  CircularProgress,
  Alert,
  Paper,
  Grid,
  List,
  ListItem,
  ListItemText,
  Chip 
} from '@mui/material';
import { getModelInfo } from '../services/api';

const ModelInfo = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const info = await getModelInfo();
        setModelInfo(info);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchModelInfo();
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ mt: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Model Information
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Model Configuration
              </Typography>
              
              <List>
                <ListItem>
                  <ListItemText
                    primary="Model Name"
                    secondary={modelInfo.model_name}
                  />
                </ListItem>
                <Divider component="li" />
                
                <ListItem>
                  <ListItemText
                    primary="Ollama URL"
                    secondary={modelInfo.ollama_url}
                  />
                </ListItem>
                <Divider component="li" />
                
                <ListItem>
                  <ListItemText
                    primary="Classifier Status"
                    secondary={
                      <Chip
                        label={modelInfo.classifier_trained ? 'Trained' : 'Not Trained'}
                        color={modelInfo.classifier_trained ? 'success' : 'warning'}
                        size="small"
                      />
                    }
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Available Models
              </Typography>
              
              {modelInfo.available_models && modelInfo.available_models.length > 0 ? (
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.100', maxHeight: 300, overflow: 'auto' }}>
                  <List dense>
                    {modelInfo.available_models.map((model, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={model}
                          secondary={model === modelInfo.model_name ? 'Currently in use' : ''}
                        />
                        {model === modelInfo.model_name && (
                          <Chip label="Active" color="primary" size="small" />
                        )}
                      </ListItem>
                    ))}
                  </List>
                </Paper>
              ) : (
                <Alert severity="warning">
                  Could not fetch available models from Ollama. Make sure Ollama is running.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {modelInfo.categories && modelInfo.categories.length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Known Categories
                </Typography>
                
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {modelInfo.categories.map((category, index) => (
                    <Chip key={index} label={category} color="primary" variant="outlined" />
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default ModelInfo;