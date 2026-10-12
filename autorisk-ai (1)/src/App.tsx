/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { StatCard } from './components/StatCard';
import { ApplicationForm } from './components/ApplicationForm';
import { RiskResultsPanel } from './components/RiskResultsPanel';
import {
  INITIAL_STAT_METRICS,
  EMPTY_APPLICATION_FORM,
  SAMPLE_PROFILES,
} from './data/mockData';
import {
  SampleProfileId,
  VehicleApplicationFormValues,
  FormValidationErrors,
  MockRiskAssessment,
  RiskCategory,
  ContributingFactor,
} from './types';

function validateApplicationForm(
  values: VehicleApplicationFormValues
): FormValidationErrors {
  const errors: FormValidationErrors = {};

  if (!values.applicantName.trim()) {
    errors.applicantName = 'Policyholder full name is required.';
  } else if (values.applicantName.trim().length < 2) {
    errors.applicantName = 'Enter at least 2 characters for the name.';
  }

  if (values.vehicleAge.trim() === '') {
    errors.vehicleAge = 'Vehicle age is required.';
  } else {
    const vAge = Number(values.vehicleAge);
    if (!Number.isInteger(vAge) || vAge < 0 || vAge > 40) {
      errors.vehicleAge = 'Enter a whole number between 0 and 40 years.';
    }
  }

  if (!values.vehicleType.trim()) {
    errors.vehicleType = 'Please select a vehicle type.';
  }

  if (!values.vehicleUsage.trim()) {
    errors.vehicleUsage = 'Please select primary vehicle usage.';
  }

  if (values.annualMileage.trim() === '') {
    errors.annualMileage = 'Annual mileage is required.';
  } else {
    const miles = Number(values.annualMileage);
    if (Number.isNaN(miles) || miles < 500 || miles > 150000) {
      errors.annualMileage = 'Enter annual mileage between 500 and 150,000.';
    }
  }

  if (values.driverAge.trim() === '') {
    errors.driverAge = 'Driver age is required.';
  } else {
    const dAge = Number(values.driverAge);
    if (!Number.isInteger(dAge) || dAge < 16 || dAge > 100) {
      errors.driverAge = 'Enter a whole age between 16 and 100 years.';
    }
  }

  if (values.driverExperience.trim() === '') {
    errors.driverExperience = 'Driver experience is required.';
  } else {
    const exp = Number(values.driverExperience);
    const dAge = Number(values.driverAge);
    if (!Number.isInteger(exp) || exp < 0 || exp > 80) {
      errors.driverExperience = 'Enter whole years between 0 and 80.';
    } else if (!errors.driverAge && exp > dAge - 15) {
      errors.driverExperience = `Cannot exceed ${Math.max(0, dAge - 15)} yrs for age ${dAge}.`;
    }
  }

  if (values.previousClaims.trim() === '') {
    errors.previousClaims = 'Previous claims count is required.';
  } else {
    const prev = Number(values.previousClaims);
    if (!Number.isInteger(prev) || prev < 0 || prev > 20) {
      errors.previousClaims = 'Enter a whole number between 0 and 20.';
    }
  }

  if (values.atFaultClaims.trim() === '') {
    errors.atFaultClaims = 'At-fault claims count is required.';
  } else {
    const atFault = Number(values.atFaultClaims);
    const prev = Number(values.previousClaims || 0);
    if (!Number.isInteger(atFault) || atFault < 0 || atFault > 20) {
      errors.atFaultClaims = 'Enter a whole number between 0 and 20.';
    } else if (!errors.previousClaims && atFault > prev) {
      errors.atFaultClaims = 'At-fault claims cannot exceed previous claims.';
    }
  }

  return errors;
}

/**
 * Deterministic demo scoring function (0–100) and template-based underwriting summary.
 * Classifies score into Low (0–33), Medium (34–66), or High (67–100).
 */
function calculateDeterministicRisk(
  values: VehicleApplicationFormValues
): MockRiskAssessment {
  const age = Number(values.driverAge) || 35;
  const experience = Number(values.driverExperience) || 10;
  const vehicleAge = Number(values.vehicleAge) || 0;
  const prevClaims = Number(values.previousClaims) || 0;
  const atFaultClaims = Number(values.atFaultClaims) || 0;
  const mileage = Number(values.annualMileage) || 10000;
  const deductible = Number(values.deductibleAmount) || 1000;

  let rawScore = 15; // Base demo points
  const factors: ContributingFactor[] = [];

  // 1. Driver Experience & Age Factor
  if (experience < 3) {
    rawScore += 18;
    factors.push({
      factor: `Limited driving experience (${experience} yrs)`,
      impact: '+18 pts',
    });
  } else if (experience < 7) {
    rawScore += 8;
    factors.push({
      factor: `Moderate driving experience (${experience} yrs)`,
      impact: '+8 pts',
    });
  } else if (experience >= 15) {
    rawScore -= 5;
    factors.push({
      factor: `Seasoned driver history (${experience} yrs experience)`,
      impact: '-5 pts',
    });
  }

  if (age < 25) {
    rawScore += 12;
    factors.push({
      factor: `Younger driver age bracket (${age} yrs)`,
      impact: '+12 pts',
    });
  } else if (age > 70) {
    rawScore += 8;
    factors.push({
      factor: `Senior driver age bracket (${age} yrs)`,
      impact: '+8 pts',
    });
  }

  // 2. Claims History Factor
  const nonFaultClaims = Math.max(0, prevClaims - atFaultClaims);
  if (atFaultClaims > 0) {
    const atFaultPoints = atFaultClaims * 18;
    rawScore += atFaultPoints;
    factors.push({
      factor: `At-fault claims recorded (${atFaultClaims})`,
      impact: `+${atFaultPoints} pts`,
    });
  }
  if (nonFaultClaims > 0) {
    const nonFaultPoints = nonFaultClaims * 7;
    rawScore += nonFaultPoints;
    factors.push({
      factor: `Prior non-fault claims (${nonFaultClaims})`,
      impact: `+${nonFaultPoints} pts`,
    });
  }
  if (prevClaims === 0) {
    factors.push({
      factor: 'Clean prior claims record (0 claims)',
      impact: '+0 pts',
    });
  }

  // 3. Annual Mileage & Usage Factor
  if (mileage > 20000) {
    rawScore += 15;
    factors.push({
      factor: `High annual mileage exposure (${mileage.toLocaleString()} mi/yr)`,
      impact: '+15 pts',
    });
  } else if (mileage > 12000) {
    rawScore += 8;
    factors.push({
      factor: `Above-average annual mileage (${mileage.toLocaleString()} mi/yr)`,
      impact: '+8 pts',
    });
  } else {
    factors.push({
      factor: `Standard annual mileage (${mileage.toLocaleString()} mi/yr)`,
      impact: '+0 pts',
    });
  }

  if (values.vehicleUsage === 'Rideshare / Delivery') {
    rawScore += 16;
    factors.push({
      factor: 'High-exposure usage (Rideshare / Delivery)',
      impact: '+16 pts',
    });
  } else if (values.vehicleUsage === 'Business / Commercial') {
    rawScore += 10;
    factors.push({
      factor: 'Commercial / Business vehicle usage',
      impact: '+10 pts',
    });
  } else if (values.vehicleUsage === 'Personal Commute') {
    rawScore += 4;
    factors.push({
      factor: 'Daily personal commute usage',
      impact: '+4 pts',
    });
  } else if (values.vehicleUsage === 'Pleasure / Weekend') {
    factors.push({
      factor: 'Low-exposure pleasure / weekend usage',
      impact: '+0 pts',
    });
  }

  // 4. Vehicle Type & Vehicle Age Factor
  if (values.vehicleType === 'Performance') {
    rawScore += 14;
    factors.push({
      factor: 'Performance / Coupe vehicle classification',
      impact: '+14 pts',
    });
  } else if (values.vehicleType === 'EV') {
    rawScore += 6;
    factors.push({
      factor: 'Specialized EV repair profile',
      impact: '+6 pts',
    });
  } else if (values.vehicleType === 'Van' || values.vehicleType === 'Truck') {
    rawScore += 5;
    factors.push({
      factor: `${values.vehicleType} chassis weight & utility profile`,
      impact: '+5 pts',
    });
  }

  if (vehicleAge > 12) {
    rawScore += 9;
    factors.push({
      factor: `Older vehicle age (${vehicleAge} yrs)`,
      impact: '+9 pts',
    });
  } else if (vehicleAge <= 3) {
    factors.push({
      factor: `Late-model vehicle safety equipment (${vehicleAge} yrs old)`,
      impact: '+0 pts',
    });
  }

  // Clamp score strictly between 0 and 100
  const riskIndexScore = Math.min(100, Math.max(0, Math.round(rawScore)));

  // Classify strictly into Low (0–33), Medium (34–66), or High (67–100)
  let riskCategory: RiskCategory = 'Low';
  let categoryRangeLabel = '0–33';
  if (riskIndexScore >= 67) {
    riskCategory = 'High';
    categoryRangeLabel = '67–100';
  } else if (riskIndexScore >= 34) {
    riskCategory = 'Medium';
    categoryRangeLabel = '34–66';
  }

  // Deterministic reference code derived from applicant & vehicle inputs
  const nameSeed = values.applicantName
    .trim()
    .split('')
    .reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
  const deterministicNum =
    100000 +
    ((nameSeed * 37 + riskIndexScore * 101 + mileage + vehicleAge * 19) % 899999);

  // Illustrative annual premium calculation
  const basePremium = 960;
  const deductibleFactor =
    deductible === 500 ? 1.14 : deductible === 2500 ? 0.88 : 1.0;
  const estimatedAnnualPremium = Math.round(
    (basePremium + riskIndexScore * 15 + prevClaims * 120 + atFaultClaims * 240) *
      deductibleFactor
  );

  // Deterministic Underwriting Summary Template
  const makeModelDisplay = values.vehicleMakeModel.trim()
    ? `${values.vehicleMakeModel.trim()} (${values.vehicleType})`
    : `unspecified ${values.vehicleType}`;

  const underwritingSummary = `Applicant ${values.applicantName.trim()} (age ${age}, ${experience} yrs driving experience in ${values.policyState}) submitted an application for a ${vehicleAge}-year-old ${makeModelDisplay} operated for ${values.vehicleUsage.toLowerCase()} at ${mileage.toLocaleString()} mi/yr. Based on ${prevClaims} total prior claim(s) (${atFaultClaims} at-fault) and selected ${values.coverageTier} coverage ($${Number(values.deductibleAmount).toLocaleString()} deductible), the deterministic model assigns an illustrative score of ${riskIndexScore}/100 (${riskCategory} Risk category).`;

  // Deterministic Recommended Review Actions
  const recommendedActions: string[] = [];
  if (riskCategory === 'Low') {
    recommendedActions.push(
      'Eligible for standard rate tier validation and automated quote issuance.'
    );
  } else if (riskCategory === 'Medium') {
    recommendedActions.push(
      'Apply standard tier surcharge multiplier and verify prior carrier loss history.'
    );
  } else {
    recommendedActions.push(
      'Escalate file to Senior Underwriter for manual risk review before binding coverage.'
    );
  }

  if (atFaultClaims > 0) {
    recommendedActions.push(
      `Order CLUE loss-history report to confirm details of ${atFaultClaims} at-fault claim(s).`
    );
  }
  if (mileage > 15000) {
    recommendedActions.push(
      `Request annual odometer verification for high mileage (${mileage.toLocaleString()} mi/yr).`
    );
  }
  if (
    values.vehicleUsage === 'Rideshare / Delivery' ||
    values.vehicleUsage === 'Business / Commercial'
  ) {
    recommendedActions.push(
      `Confirm commercial or rideshare endorsement eligibility for ${values.vehicleUsage}.`
    );
  }

  // Flagged Missing Information based on entered values
  const missingInformationFlags: string[] = [];
  if (!values.vehicleMakeModel.trim()) {
    missingInformationFlags.push(
      'Vehicle Make & Model was left blank on the application form.'
    );
  }
  if (prevClaims > 0) {
    missingInformationFlags.push(
      'Incident dates and claim payout amounts are not attached for reported prior claims.'
    );
  }
  if (vehicleAge > 10) {
    missingInformationFlags.push(
      `Physical vehicle inspection & safety certification required for ${vehicleAge}-year-old vehicle.`
    );
  }
  missingInformationFlags.push(
    '17-character Vehicle Identification Number (VIN) and Motor Vehicle Record (MVR) license number not yet provided.'
  );

  return {
    referenceId: `DEMO-${deterministicNum}`,
    riskIndexScore,
    riskCategory,
    categoryRangeLabel,
    estimatedAnnualPremium,
    contributingFactors: factors.slice(0, 6),
    underwritingSummary,
    recommendedActions,
    missingInformationFlags,
  };
}

export default function App() {
  const [selectedSampleId, setSelectedSampleId] = useState<SampleProfileId | null>(null);
  const [formValues, setFormValues] = useState<VehicleApplicationFormValues>(
    EMPTY_APPLICATION_FORM
  );
  const [formErrors, setFormErrors] = useState<FormValidationErrors>({});
  const [assessment, setAssessment] = useState<MockRiskAssessment | null>(null);
  const [statusBanner, setStatusBanner] = useState<string | null>(null);

  const handleFieldChange = (
    field: keyof VehicleApplicationFormValues,
    value: string
  ) => {
    setSelectedSampleId(null);
    setFormValues((prev) => ({ ...prev, [field]: value }));
    if (formErrors[field]) {
      setFormErrors((prev) => {
        const next = { ...prev };
        delete next[field];
        return next;
      });
    }
  };

  // Clicking a sample fills the form so the user can click Analyze Risk to view its results
  const handleSelectSample = (sampleId: SampleProfileId) => {
    const preset = SAMPLE_PROFILES[sampleId];
    setSelectedSampleId(sampleId);
    setFormValues(preset.values);
    setFormErrors({});
    setStatusBanner(
      `${preset.label} (${preset.values.applicantName}) loaded into form. Click "Analyze Risk" to view results.`
    );
  };

  const handleResetAll = () => {
    setSelectedSampleId(null);
    setFormValues(EMPTY_APPLICATION_FORM);
    setFormErrors({});
    setAssessment(null);
    setStatusBanner('Form fields and risk assessment results cleared.');
  };

  const handleSubmitForm = (e: React.FormEvent) => {
    e.preventDefault();
    const validationErrors = validateApplicationForm(formValues);
    if (Object.keys(validationErrors).length > 0) {
      setFormErrors(validationErrors);
      setStatusBanner('Please resolve the highlighted validation errors before analyzing risk.');
      return;
    }

    setFormErrors({});
    const result = calculateDeterministicRisk(formValues);
    setAssessment(result);
    setStatusBanner(
      `Risk analysis completed for ${formValues.applicantName.trim()}: ${result.riskIndexScore}/100 (${result.riskCategory} Risk).`
    );
  };

  return (
    <div className="min-h-screen flex flex-col lg:flex-row bg-white text-slate-900">
      {/* Dark Navy Sidebar */}
      <Sidebar
        selectedSampleId={selectedSampleId}
        onSelectSample={handleSelectSample}
        onResetAll={handleResetAll}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-white">
        <Header
          selectedSampleId={selectedSampleId}
          onSelectSample={handleSelectSample}
          onResetAll={handleResetAll}
        />

        <main className="flex-1 px-4 sm:px-8 py-6 sm:py-8 max-w-[1360px] w-full mx-auto space-y-6 sm:space-y-8">
          {/* Page Intro Header */}
          <div className="flex flex-wrap items-baseline justify-between gap-4">
            <div>
              <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
                Vehicle Underwriting &amp; Risk Analytics
              </h1>
              <p className="text-sm text-slate-500 mt-1">
                Review portfolio risk metrics, capture vehicle application data, and inspect deterministic underwriting results.
              </p>
            </div>
            {statusBanner && (
              <div
                role="status"
                className="text-xs font-medium text-blue-800 bg-blue-50 border border-blue-200 px-3.5 py-2 rounded-lg"
              >
                {statusBanner}
              </div>
            )}
          </div>

          {/* Four Statistic Cards */}
          <section aria-label="Key Insurance Portfolio Metrics">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
              {INITIAL_STAT_METRICS.map((metric) => (
                <StatCard key={metric.id} metric={metric} />
              ))}
            </div>
          </section>

          {/* Two-Column Workbench: Application Form + Risk Results Panel */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            <div className="lg:col-span-7">
              <ApplicationForm
                values={formValues}
                errors={formErrors}
                selectedSampleId={selectedSampleId}
                onChange={handleFieldChange}
                onSubmit={handleSubmitForm}
                onReset={handleResetAll}
                onSelectSample={handleSelectSample}
              />
            </div>

            <div className="lg:col-span-5">
              <RiskResultsPanel
                assessment={assessment}
                onClearAssessment={() => {
                  setAssessment(null);
                  setStatusBanner('Risk results panel cleared.');
                }}
                onSelectSample={handleSelectSample}
              />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
