export type RiskCategory = 'Low' | 'Medium' | 'High';

export type SampleProfileId = 'low' | 'medium' | 'high';

export interface StatMetric {
  id: string;
  label: string;
  value: string;
  change: string;
  context: string;
  trend: 'positive' | 'neutral' | 'attention';
}

export interface VehicleApplicationFormValues {
  applicantName: string;
  policyState: string;
  driverAge: string;
  driverExperience: string;
  vehicleAge: string;
  vehicleMakeModel: string;
  vehicleType: string;
  vehicleUsage: string;
  annualMileage: string;
  previousClaims: string;
  atFaultClaims: string;
  coverageTier: string;
  deductibleAmount: string;
}

export interface SampleProfilePreset {
  id: SampleProfileId;
  label: string;
  shortTag: string;
  description: string;
  expectedCategory: RiskCategory;
  values: VehicleApplicationFormValues;
}

export type FormValidationErrors = Partial<Record<keyof VehicleApplicationFormValues, string>>;

export interface ContributingFactor {
  factor: string;
  impact: string;
}

export interface MockRiskAssessment {
  referenceId: string;
  riskIndexScore: number;
  riskCategory: RiskCategory;
  categoryRangeLabel: string;
  estimatedAnnualPremium: number;
  contributingFactors: ContributingFactor[];
  underwritingSummary: string;
  recommendedActions: string[];
  missingInformationFlags: string[];
}
